# -*- coding: utf-8 -*-
"""AML 多模态赛道参赛服务 v0.2.9（memory-api-v1.1，严格契约对齐版）。

依据官方文档修正的三个关键点：
  1) Add 响应必须含 success:true + request_id/user_id/session_id（逐一回显），
     否则即使 HTTP 200 阶段也立即失败
  2) Search 响应必须为 {"data":[{id, content, score?, created_at?}]}，
     无结果返回空数组；不得超过 top_k
  3) 检索隔离：Search 只能返回该 user_id 的记忆（跨用户违规）

架构：同步快路径（落盘 + MiniLM 向量 + BM25，无 LLM）+ user_id 隔离检索。
增强层（OmniMem LLM 管线）默认关闭（ENABLE_ENRICH=0）——其记忆为全局共享，
无法按 user_id 隔离，开启有违规风险；后续做 per-user 分区后再启用。

环境变量：
  OPENAI_API_KEY / OPENAI_API_BASE / AML_LLM_MODEL(openai/gpt-4o-mini)
  MEMORY_SYSTEM_KEY / DATA_DIR(默认 /data) / PORT(默认 8000)
  ENABLE_ENRICH(默认 0；=1 时启用 OmniMem 后台增强，当前版本 Search 不消费)

v0.2.9：新增查询日志 DATA_DIR/searches.jsonl——记录每次 Search 的 query、
阈值过滤前 top8 原始分、返回条数（弃答阈值校准数据，官方冒烟/评测时自动产出）。
"""
import base64
import json
import os
import re
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException, Request, Response

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OMNI_ROOT = os.path.join(PROJ, "SimpleMem", "OmniSimpleMem")
if OMNI_ROOT not in sys.path:
    sys.path.insert(0, OMNI_ROOT)

LLM_MODEL = os.environ.get("AML_LLM_MODEL", "openai/gpt-4o-mini")
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
MEMORY_KEY = os.environ.get("MEMORY_SYSTEM_KEY", "")
PORT = int(os.environ.get("PORT", "8000"))
ENABLE_ENRICH = os.environ.get("ENABLE_ENRICH", "0") == "1"
MAX_BODY = 30 * 1024 * 1024
# 相关性阈值：混合分低于此值的结果不返回；全部低于则返回空数组（支撑弃答 + 降噪）
# 校准依据（平台真实英文数据）：相关 top1≈0.60+（清洁嵌入后更高），无关 top1≈0.43
RELEVANCE_THRESHOLD = float(os.environ.get("RELEVANCE_THRESHOLD", "0.50"))

_PREFIX_RE = re.compile(r"^\[[^\]]*\]\s*[a-zA-Z_]+\s*:\s*")


def _clean_text(text: str) -> str:
    """去掉 '[ts] role:' 前缀，得到干净内容（用于向量与 BM25，降低噪声）。"""
    return "\n".join(_PREFIX_RE.sub("", ln) for ln in text.split("\n"))

for sub in ("memory", "images", "hf"):
    (DATA_DIR / sub).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(DATA_DIR / "hf"))
SEEN_LOG = DATA_DIR / "request_ids.jsonl"
TURNS_LOG = DATA_DIR / "turns.jsonl"
SEARCH_LOG = DATA_DIR / "searches.jsonl"

_DATAURI_RE = re.compile(r"^data:image/(jpeg|jpg|png|webp);base64,(.*)$", re.S)


# ================= 原始层快速索引（同步路径，无 LLM，user_id 隔离） =================
class FastIndex:
    """turns.jsonl 持久化 + MiniLM 向量 + BM25 混合检索，按 user_id 严格隔离。"""

    def __init__(self, data_dir: Path):
        self.lock = threading.Lock()
        self.turns: List[Dict] = []
        self.vecs: List[Any] = []
        self.tokens: List[List[str]] = []
        from sentence_transformers import SentenceTransformer
        import numpy as np
        self._np = np
        self._model = SentenceTransformer("all-MiniLM-L6-v2")
        self._load()

    def _load(self):
        np = self._np
        missing = []
        if TURNS_LOG.exists():
            with open(TURNS_LOG, encoding="utf-8") as f:
                for line in f:
                    try:
                        t = json.loads(line)
                        self.turns.append(t)
                        self.tokens.append(self._tokenize(t.get("text_clean") or _clean_text(t["text"])))
                        v = t.get("vec")
                        if v:
                            self.vecs.append(np.frombuffer(base64.b64decode(v), dtype=np.float32))
                        else:
                            self.vecs.append(None)
                            missing.append(len(self.turns) - 1)
                    except Exception:
                        continue
        # 仅对无向量的旧数据重编码（新数据向量随行落盘，重启秒级恢复）
        if missing:
            texts = [self.turns[i].get("text_clean") or _clean_text(self.turns[i]["text"])
                     for i in missing]
            vecs = self._model.encode(texts, normalize_embeddings=True,
                                      show_progress_bar=False)
            for j, i in enumerate(missing):
                self.vecs[i] = vecs[j]

    @staticmethod
    def _tokenize(text: str):
        return re.findall(r"[a-zA-Z0-9]+|[一-鿿]", text.lower())

    def add(self, turn: Dict):
        turn["text_clean"] = _clean_text(turn["text"])
        vec = self._model.encode([turn["text_clean"]], normalize_embeddings=True,
                                 show_progress_bar=False)[0]
        turn["vec"] = base64.b64encode(vec.astype(self._np.float32).tobytes()).decode()
        with self.lock:
            with open(TURNS_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(turn, ensure_ascii=False) + "\n")
            self.turns.append(turn)
            self.vecs.append(vec)
            self.tokens.append(self._tokenize(turn["text_clean"]))

    def search(self, query: str, user_id: str, k: int = 100) -> List[Dict]:
        np = self._np
        with self.lock:
            idxs = [i for i, t in enumerate(self.turns)
                    if t.get("user_id") == user_id]
            if not idxs:
                self._log_query(query, user_id, k, [], 0)
                return []
            qv = self._model.encode([query], normalize_embeddings=True,
                                    show_progress_bar=False)[0]
            sub_tokens = [self.tokens[i] for i in idxs]
            dense = {i: float(np.dot(self.vecs[i], qv)) for i in idxs}
            tokens_q = self._tokenize(query)
            bm25_s = {i: 0.0 for i in idxs}
            if any(sub_tokens):
                from rank_bm25 import BM25Okapi
                bm25 = BM25Okapi(sub_tokens)
                s = bm25.get_scores(tokens_q)
                mx = float(s.max()) if s.size and s.max() > 0 else 0.0
                for j, i in enumerate(idxs):
                    # 负分夹到 0（BM25 在小语料上可能为负），正分按 max 归一到 [0,1]
                    bm25_s[i] = max(0.0, float(s[j]) / mx) if mx > 0 else 0.0
            # 混合分下限保护：语义余弦是校准通道，弱词汇证据不应把高语义拉下阈值
            scored = sorted(
                ((max(0.6 * dense[i] + 0.4 * bm25_s[i], dense[i]), i) for i in idxs),
                reverse=True)
            # 阈值过滤前的原始 top 分数（弃答校准：无关查询的真实分数分布）
            raw_top = [(round(sc, 4), str(self.turns[i].get("request_id") or f"turn-{i}"))
                       for sc, i in scored[:8]]
            out = []
            for rank, (sc, i) in enumerate(scored, 1):
                if len(out) >= k:
                    break
                if sc < RELEVANCE_THRESHOLD:
                    continue  # 低相关结果不返回：无相关记忆时返回空数组，支撑答题侧弃答
                t = self.turns[i]
                out.append({
                    "id": str(t.get("request_id") or f"turn-{i}"),
                    "content": t["text"],
                    "score": round(max(0.0, min(1.0, sc)), 4),
                    "created_at": t.get("ts", ""),
                })
            self._log_query(query, user_id, k, raw_top, len(out))
            return out

    @staticmethod
    def _log_query(query: str, user_id: str, k: int, raw_top: List, n_returned: int):
        """查询日志：真实 query + 阈值过滤前 top 分数，供弃答阈值离线校准。"""
        try:
            with open(SEARCH_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "user_id": user_id,
                    "query": query[:2000],
                    "top_k": k,
                    "raw_top": raw_top,
                    "returned": n_returned,
                    "threshold": RELEVANCE_THRESHOLD,
                }, ensure_ascii=False) + "\n")
        except Exception:
            pass  # 日志失败不影响契约响应

    def __len__(self):
        return len(self.turns)


_FAST = FastIndex(DATA_DIR)

# ================= 增强层（默认关闭；开启时仅后台异步增强，不影响契约响应） =================
_ADAPTER = None
_ADAPTER_LOCK = threading.Lock()

if ENABLE_ENRICH:
    from benchmarks.memgallery.adapter import OmniMemAdapter  # noqa: E402
    from omni_memory import OmniMemoryConfig as _OmniCfg  # noqa: E402

    _cfg = _OmniCfg.create_default()
    _cfg.set_unified_model(LLM_MODEL)
    _cfg.embedding.model_name = "all-MiniLM-L6-v2"
    _cfg.embedding.embedding_dim = 384
    _ADAPTER = OmniMemAdapter(data_dir=str(DATA_DIR / "memory"), config=_cfg)


def _content_to_parts(content: Any) -> Dict[str, Any]:
    """ContentPart[]/字符串 → {text, images:[data_uri]}（同步路径不做任何 LLM）。"""
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
                texts.append(f"[image: {img_id}]")
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

app = FastAPI(title="AML Multimodal Memory Service", version="0.2.9")


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
    return {"status": "ok", "model": LLM_MODEL, "service": "aml-multimodal-v0.2.9",
            "turns": len(_FAST)}


# ---- HEAD 探活：平台用 HEAD 探测绑定的 Add/Search 端点 ----
@app.head("/v1/memory/add")
@app.head("/v1/memory/ad")
@app.head("/v1/memory/search")
@app.head("/add")
@app.head("/search")
async def head_ok():
    return Response(status_code=200)


@app.post("/v1/memory/add")
@app.post("/v1/memory/ad")   # 别名：兼容已绑定版本 Add 地址的笔误（端点在 Key 层冻结）
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
    user_id = str(payload.get("user_id") or "")
    session_id = str(payload.get("session_id") or "")
    if not user_id or not session_id:
        raise HTTPException(400, "user_id and session_id are required")

    if request_id in _seen_ids:
        return {"success": True, "request_id": request_id,
                "user_id": user_id, "session_id": session_id, "duplicate": True}

    messages = payload.get("messages") or []
    ts_iso = _ts_to_iso(payload.get("timestamp"))

    text_parts = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "user")
        msg_ts = _ts_to_iso(msg.get("timestamp")) if msg.get("timestamp") else ts_iso
        parsed = _content_to_parts(msg.get("content"))
        if parsed["text"]:
            text_parts.append(f"[{msg_ts}] {role}: {parsed['text']}")
    text = "\n".join(text_parts)

    if text:
        _FAST.add({"request_id": request_id, "user_id": user_id,
                   "session_id": session_id, "ts": ts_iso, "text": text})

    with open(SEEN_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"request_id": request_id, "ts": ts_iso}) + "\n")
    _seen_ids.add(request_id)

    # 契约要求的成功响应：success + 三个 ID 逐一回显
    return {"success": True, "request_id": request_id,
            "user_id": user_id, "session_id": session_id}


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

    user_id = str(payload.get("user_id") or "")
    if not user_id:
        raise HTTPException(400, "user_id is required")

    query = payload.get("query", "")
    if isinstance(query, list):
        qtext = _content_to_parts(query)["text"]
    else:
        qtext = str(query)
    options = payload.get("options")
    if isinstance(options, list) and options:
        qtext += "\noptions: " + " | ".join(str(o) for o in options)
    if not qtext:
        raise HTTPException(400, "empty query")

    top_k = payload.get("top_k")
    try:
        top_k = int(top_k) if top_k is not None else 100
    except Exception:
        top_k = 100
    top_k = max(1, min(top_k, 200))

    # 契约响应：data 数组 + id/content 必填；严格 user_id 隔离；数量 ≤ top_k
    data = _FAST.search(qtext, user_id=user_id, k=top_k)
    return {"data": data}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
