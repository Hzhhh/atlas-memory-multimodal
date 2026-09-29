# -*- coding: utf-8 -*-
"""AML 多模态赛道参赛服务 v0.2（memory-api-v1.1）。

v0.2 架构（针对官方评测 16 并发长跑优化，教训来自文本赛道 Full 失败诊断）：
  同步 Add 快路径（<100ms 量级，无 LLM）：
    原始轮次落盘(turns.jsonl) + MiniLM 嵌入入 FAISS + BM25 脏标记
    → 返回 200 时内容已持久化且可检索（契约合规）
  后台增强工作线程（LLM=gpt-4o-mini，官方要求）：
    OmniMemAdapter.store 全管线（摘要/实体/图谱）异步补全；图像 caption 异步补全
  Search：
    OmniMem 增强层 recall + 原始层 BM25/向量混合检索合并
    → 无论增强进度如何，原始内容永远可检索

环境变量：
  OPENAI_API_KEY / OPENAI_API_BASE / AML_LLM_MODEL(openai/gpt-4o-mini)
  MEMORY_SYSTEM_KEY / DATA_DIR(默认 /data) / PORT(默认 8000)
  ENRICH_WORKERS(默认 4)
"""
import base64
import concurrent.futures
import json
import os
import queue
import re
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException, Request

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OMNI_ROOT = os.path.join(PROJ, "SimpleMem", "OmniSimpleMem")
if OMNI_ROOT not in sys.path:
    sys.path.insert(0, OMNI_ROOT)

LLM_MODEL = os.environ.get("AML_LLM_MODEL", "openai/gpt-4o-mini")
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
MEMORY_KEY = os.environ.get("MEMORY_SYSTEM_KEY", "")
PORT = int(os.environ.get("PORT", "8000"))
ENRICH_WORKERS = int(os.environ.get("ENRICH_WORKERS", "4"))
MAX_BODY = 30 * 1024 * 1024

for sub in ("memory", "images", "hf"):
    (DATA_DIR / sub).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(DATA_DIR / "hf"))
USAGE_LOG = DATA_DIR / "token_usage.jsonl"
SEEN_LOG = DATA_DIR / "request_ids.jsonl"
TURNS_LOG = DATA_DIR / "turns.jsonl"


# ---- token 记账 ----
def _install_usage_accounting() -> None:
    try:
        from openai.resources.chat.completions import Completions
        _orig = Completions.create

        def _create(self, *a, **kw):
            resp = _orig(self, *a, **kw)
            try:
                u = getattr(resp, "usage", None)
                if u is not None:
                    with open(USAGE_LOG, "a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "ts": datetime.now().isoformat(timespec="seconds"),
                            "model": getattr(resp, "model", kw.get("model", "?")),
                            "in": u.prompt_tokens or 0,
                            "out": u.completion_tokens or 0,
                        }) + "\n")
            except Exception:
                pass
            return resp

        Completions.create = _create
    except Exception:
        pass


_install_usage_accounting()
from openai import OpenAI  # noqa: E402

_client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY", ""),
    base_url=os.environ.get("OPENAI_API_BASE", "https://openrouter.ai/api/v1"),
)

_DATAURI_RE = re.compile(r"^data:image/(jpeg|jpg|png|webp);base64,(.*)$", re.S)


# ================= 原始层快速索引（同步路径，无 LLM） =================
class FastIndex:
    """turns.jsonl 持久化 + MiniLM/FAISS 向量 + BM25 词法混合检索。"""

    def __init__(self, data_dir: Path):
        self.dir = data_dir
        self.lock = threading.Lock()
        self.turns: List[Dict] = []
        self._bm25 = None
        self._bm25_dirty = True
        from sentence_transformers import SentenceTransformer
        import faiss
        import numpy as np
        self._np = np
        self._faiss_mod = faiss
        self._model = SentenceTransformer("all-MiniLM-L6-v2")
        self._index = faiss.IndexFlatIP(384)
        self._load()

    def _load(self):
        if TURNS_LOG.exists():
            with open(TURNS_LOG, encoding="utf-8") as f:
                for line in f:
                    try:
                        t = json.loads(line)
                        self.turns.append(t)
                    except Exception:
                        continue
        if self.turns:
            vecs = self._model.encode([t["text"] for t in self.turns],
                                      normalize_embeddings=True, show_progress_bar=False)
            self._index.add(self._np.asarray(vecs, dtype="float32"))
            self._bm25_dirty = True

    def add(self, turn: Dict):
        vec = self._model.encode([turn["text"]], normalize_embeddings=True,
                                 show_progress_bar=False)
        with self.lock:
            with open(TURNS_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(turn, ensure_ascii=False) + "\n")
            self.turns.append(turn)
            self._index.add(self._np.asarray(vec, dtype="float32"))
            self._bm25_dirty = True

    def _ensure_bm25(self):
        if self._bm25_dirty or self._bm25 is None:
            from rank_bm25 import BM25Okapi
            corpus = [self._tokenize(t["text"]) for t in self.turns]
            self._bm25 = BM25Okapi(corpus) if corpus else None
            self._bm25_dirty = False

    @staticmethod
    def _tokenize(text: str):
        return re.findall(r"[a-zA-Z0-9]+|[一-鿿]", text.lower())

    def search(self, query: str, k: int = 30) -> List[Dict]:
        with self.lock:
            self._ensure_bm25()
            n = len(self.turns)
            if n == 0:
                return []
            qv = self._model.encode([query], normalize_embeddings=True,
                                    show_progress_bar=False)
            _, dense_idx = self._index.search(self._np.asarray(qv, dtype="float32"),
                                              min(n, max(k, 20)))
            bm25_scores = (self._bm25.get_scores(self._tokenize(query))
                           if self._bm25 is not None
                           else self._np.zeros(n))
            mx = float(bm25_scores.max()) if bm25_scores.size and bm25_scores.max() > 0 else 1.0
            cand = set(int(i) for i in dense_idx[0] if 0 <= i < n)
            top_bm = sorted(range(n), key=lambda i: -bm25_scores[i])[:max(k, 20)]
            cand.update(top_bm)
            scored = []
            for i in cand:
                hybrid = 0.6 * (i in set(int(x) for x in dense_idx[0])) + \
                         0.4 * (bm25_scores[i] / mx if mx > 0 else 0.0)
                scored.append((hybrid, i))
            scored.sort(reverse=True)
            return [self.turns[i] for _, i in scored[:k]]

    def __len__(self):
        return len(self.turns)


_FAST = FastIndex(DATA_DIR)

# ================= 增强层（后台，OmniMem 全管线） =================
from benchmarks.memgallery.adapter import OmniMemAdapter  # noqa: E402
from omni_memory import OmniMemoryConfig  # noqa: E402


def _build_memory():
    cfg = OmniMemoryConfig.create_default()
    cfg.set_unified_model(LLM_MODEL)
    cfg.embedding.model_name = "all-MiniLM-L6-v2"
    cfg.embedding.embedding_dim = 384
    return OmniMemAdapter(data_dir=str(DATA_DIR / "memory"), config=cfg)


_ADAPTER = _build_memory()
_ADAPTER_LOCK = threading.Lock()

_enrich_q: "queue.Queue[Dict]" = queue.Queue()
_ENRICHED_IDS = set()
_enrich_log = DATA_DIR / "enriched_ids.jsonl"
if _enrich_log.exists():
    with open(_enrich_log, encoding="utf-8") as f:
        for line in f:
            try:
                _ENRICHED_IDS.add(json.loads(line)["request_id"])
            except Exception:
                pass


def _enrich_worker():
    while True:
        item = _enrich_q.get()
        if item is None:
            return
        try:
            rid = item["request_id"]
            if rid in _ENRICHED_IDS:
                continue
            if item.get("caption_job"):
                # 图像 caption 补全
                info = _caption_image_llm(item["caption_job"])
                text = (f"image detail (request {rid}):\nimage_id: {info['img_id']}\n"
                        f"image_caption: {info['caption']}")
                with _ADAPTER_LOCK:
                    _ADAPTER.store({"text": text, "image": None,
                                    "timestamp": item.get("ts", ""),
                                    "dialogue_id": f"caption:{rid}"})
            else:
                with _ADAPTER_LOCK:
                    _ADAPTER.store(item["observation"])
            with open(_enrich_log, "a", encoding="utf-8") as f:
                f.write(json.dumps({"request_id": rid}) + "\n")
            _ENRICHED_IDS.add(rid)
        except Exception:
            pass  # 增强失败不影响原始层可检索性
        finally:
            _enrich_q.task_done()


for _ in range(ENRICH_WORKERS):
    threading.Thread(target=_enrich_worker, daemon=True).start()


def _caption_image_llm(data_uri: str) -> Dict[str, str]:
    m = _DATAURI_RE.match(data_uri.strip())
    if not m:
        return {"img_id": "invalid", "caption": ""}
    ext = {"jpg": "jpeg"}.get(m.group(1), m.group(1))
    raw = base64.b64decode(m.group(2))
    img_id = uuid.uuid4().hex[:16]
    (DATA_DIR / "images" / f"{img_id}.{ext}").write_bytes(raw)
    try:
        resp = _client.chat.completions.create(
            model=LLM_MODEL, temperature=0.0, max_tokens=220,
            messages=[{"role": "user", "content": [
                {"type": "text",
                 "text": "Describe this image concisely for a long-term memory system: "
                         "objects, people, visible text (OCR), scene, notable details."},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ]}],
        )
        caption = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        caption = f"(caption unavailable: {type(e).__name__})"
    return {"img_id": img_id, "caption": caption}


def _content_to_parts(content: Any) -> Dict[str, Any]:
    """ContentPart[]/字符串 → {text, images: [data_uri]}（同步路径不做任何 LLM）。"""
    if content is None:
        return {"text": "", "images": []}
    if isinstance(content, str):
        return {"text": content, "images": []}
    if not isinstance(content, list):
        raise HTTPException(400, "content must be a string or ContentPart[]")
    texts, images = [], []
    for p in content:
        if not isinstance(p, dict):
            texts.append(str(p))
            continue
        if p.get("type") == "text":
            texts.append(p.get("text", ""))
        elif p.get("type") == "image_url":
            url = (p.get("image_url") or {}).get("url", "")
            if url:
                m = _DATAURI_RE.match(url.strip())
                if not m:
                    raise HTTPException(400, "image_url must be data:image/{jpeg,png,webp};base64,")
                raw = base64.b64decode(m.group(2))
                if len(raw) > 10 * 1024 * 1024:
                    raise HTTPException(400, "decoded image exceeds 10MiB")
                img_id = uuid.uuid4().hex[:16]
                ext = {"jpg": "jpeg"}.get(m.group(1), m.group(1))
                (DATA_DIR / "images" / f"{img_id}.{ext}").write_bytes(raw)
                texts.append(f"[image: {img_id} (caption pending)]")
                images.append(url)
    return {"text": "\n".join(x for x in texts if x), "images": images}


# ================= HTTP 层 =================
_seen_ids = set()
if SEEN_LOG.exists():
    with open(SEEN_LOG, encoding="utf-8") as f:
        for line in f:
            try:
                _seen_ids.add(json.loads(line)["request_id"])
            except Exception:
                pass

app = FastAPI(title="AML Multimodal Memory Service", version="0.2.1")


def _check_auth(request: Request) -> None:
    if not MEMORY_KEY:
        return
    auth = request.headers.get("authorization", "")
    xkey = request.headers.get("x-api-key", "")
    token = ""
    if auth.lower().startswith("bearer "):
        token = auth[7:].strip()
    elif auth.lower().startswith("token "):
        token = auth[6:].strip()
    elif xkey:
        token = xkey.strip()
    if token != MEMORY_KEY:
        raise HTTPException(401, "invalid or missing memory system key")


def _ts_to_iso(ts) -> str:
    if ts is None:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        v = float(ts)
        if v > 1e12:
            v /= 1000.0
        return datetime.fromtimestamp(v, tz=timezone.utc).isoformat(timespec="seconds")
    except Exception:
        return str(ts)


@app.get("/health")
async def health():
    return {"status": "ok", "model": LLM_MODEL, "service": "aml-multimodal-v0.2",
            "turns": len(_FAST), "enriched": len(_ENRICHED_IDS)}


@app.post("/v1/memory/add")
@app.post("/add")
async def memory_add(request: Request):
    _check_auth(request)
    try:
        body = await request.body()
        if len(body) > MAX_BODY:
            raise HTTPException(413, "request body exceeds 30MiB")
        payload = json.loads(body)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"invalid JSON: {e}")

    request_id = str(payload.get("request_id") or uuid.uuid4().hex)
    if request_id in _seen_ids:
        return {"status": "ok", "request_id": request_id, "duplicate": True}

    messages = payload.get("messages") or []
    session_id = str(payload.get("session_id") or "default_session")
    ts_iso = _ts_to_iso(payload.get("timestamp"))

    text_parts, image_uris = [], []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "user")
        parsed = _content_to_parts(msg.get("content"))
        if parsed["text"]:
            text_parts.append(f"{role}: {parsed['text']}")
        image_uris.extend(parsed["images"])
    text = "\n".join(text_parts)

    turn = {"request_id": request_id, "session_id": session_id, "ts": ts_iso, "text": text}

    # —— 同步快路径：持久化 + 可检索，然后立刻 200 ——
    if text:
        _FAST.add(turn)

    with open(SEEN_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"request_id": request_id, "ts": ts_iso}) + "\n")
    _seen_ids.add(request_id)

    # —— 异步增强：OmniMem 全管线 + 图像 caption ——
    if text:
        _enrich_q.put({"request_id": request_id, "observation": {
            "text": text, "image": None, "timestamp": ts_iso,
            "dialogue_id": f"{session_id}:{request_id}"}})
    for uri in image_uris:
        _enrich_q.put({"request_id": f"{request_id}:cap{len(image_uris)}",
                       "caption_job": uri, "ts": ts_iso})

    return {"status": "ok", "request_id": request_id}


_RECALL_POOL = concurrent.futures.ThreadPoolExecutor(2, thread_name_prefix="recall")


def _adapter_recall_safe(qtext: str, lock_timeout: float = 3.0) -> str:
    if not _ADAPTER_LOCK.acquire(timeout=lock_timeout):
        return ""  # 增强层忙（后台enrichment持有锁）→ 降级原始层
    try:
        return _ADAPTER.recall(qtext) or ""
    finally:
        _ADAPTER_LOCK.release()


@app.post("/v1/memory/search")
@app.post("/search")
async def memory_search(request: Request):
    _check_auth(request)
    try:
        body = await request.body()
        if len(body) > MAX_BODY:
            raise HTTPException(413, "request body exceeds 30MiB")
        payload = json.loads(body)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"invalid JSON: {e}")

    query = payload.get("query", "")
    qtext = str(query) if not isinstance(query, list) else _content_to_parts(query)["text"]
    if not qtext:
        raise HTTPException(400, "empty query")
    top_k = int(payload.get("top_k") or 100)

    raw_turns = _FAST.search(qtext, k=min(30, max(10, top_k // 3)))

    context_parts = []
    try:
        fut = _RECALL_POOL.submit(_adapter_recall_safe, qtext)
        enriched_ctx = fut.result(timeout=10)  # 超时降级：只用原始层
        if enriched_ctx:
            context_parts.append(enriched_ctx)
    except Exception:
        pass
    if raw_turns:
        lines = [f"[{i+1}] TURN:{t['request_id']} | SESSION:{t['session_id']} | "
                 f"DATE:{t['ts']}\n{t['text']}"
                 for i, t in enumerate(raw_turns)]
        context_parts.append("\n\n".join(lines))
    context = "\n\n========\n\n".join(context_parts)

    return {"data": [{"type": "text", "text": context}], "top_k": top_k}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
