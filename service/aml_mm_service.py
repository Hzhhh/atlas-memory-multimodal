# -*- coding: utf-8 -*-
"""AML 多模态赛道参赛服务（memory-api-v1.1 契约实现）。

官方要求：
  - Add/Search 两个端点，图像以内联 base64 data URI 传入（JPEG/PNG/WebP）
  - Add 同步持久化；request_id 幂等；Add/Search 的 LLM 一律 gpt-4o-mini
  - 鉴权：Authorization: Bearer <key> / X-Api-Key: <key> / Token <key>
  - 单图解码后 ≤10MiB，单请求 ≤30MiB

架构（v0，复用本地验证过的 Omni-SimpleMem 文本模式）：
  Add:    messages[].content(str|ContentPart[]) → 文本拼接 + 图像 gpt-4o-mini 视觉
          caption 转文本 → OmniMemAdapter.store（内部做摘要/实体抽取/混合索引）
  Search: query(str|ContentPart[]) → query 图像 caption 并入 → OmniMemAdapter.recall
          → 检索上下文字符串 → {"data":[{"content": ...}]}

环境变量：
  OPENAI_API_KEY     必须（OpenRouter key）
  OPENAI_API_BASE    默认 https://openrouter.ai/api/v1
  AML_LLM_MODEL      默认 openai/gpt-4o-mini（官方硬性要求，勿改）
  MEMORY_SYSTEM_KEY  鉴权密钥（提供给平台；为空则允许匿名，仅限开发）
  DATA_DIR           默认 /data（记忆/图像/日志持久化目录）
  PORT               默认 8000
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

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OMNI_ROOT = os.path.join(PROJ, "SimpleMem", "OmniSimpleMem")
if OMNI_ROOT not in sys.path:
    sys.path.insert(0, OMNI_ROOT)

from benchmarks.memgallery.adapter import OmniMemAdapter  # noqa: E402
from omni_memory import OmniMemoryConfig  # noqa: E402

LLM_MODEL = os.environ.get("AML_LLM_MODEL", "openai/gpt-4o-mini")
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
MEMORY_KEY = os.environ.get("MEMORY_SYSTEM_KEY", "")
PORT = int(os.environ.get("PORT", "8000"))
MAX_BODY = 30 * 1024 * 1024

for sub in ("memory", "images", "hf"):
    (DATA_DIR / sub).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(DATA_DIR / "hf"))
USAGE_LOG = DATA_DIR / "token_usage.jsonl"
SEEN_LOG = DATA_DIR / "request_ids.jsonl"

# ---- token 记账（与本地一致的 monkey-patch）----
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

from openai import OpenAI  # noqa: E402  (记账补丁之后导入)

_client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY", ""),
    base_url=os.environ.get("OPENAI_API_BASE", "https://openrouter.ai/api/v1"),
)

_DATAURI_RE = re.compile(r"^data:image/(jpeg|jpg|png|webp);base64,(.*)$", re.S)


def _caption_image(data_uri: str) -> Dict[str, str]:
    """base64 图像 → 落盘 + gpt-4o-mini 视觉 caption。返回 {img_id, caption}。"""
    m = _DATAURI_RE.match(data_uri.strip())
    if not m:
        raise HTTPException(400, "image_url must be a data:image/{jpeg,png,webp};base64, URI")
    ext = {"jpg": "jpeg"}.get(m.group(1), m.group(1))
    raw = base64.b64decode(m.group(2))
    if len(raw) > 10 * 1024 * 1024:
        raise HTTPException(400, "decoded image exceeds 10MiB")
    img_id = uuid.uuid4().hex[:16]
    path = DATA_DIR / "images" / f"{img_id}.{ext}"
    path.write_bytes(raw)
    try:
        resp = _client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0.0,
            max_tokens=220,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text",
                     "text": "Describe this image concisely for a long-term memory system: "
                             "objects, people, visible text (OCR), scene, and notable details."},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }],
        )
        caption = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        caption = f"(caption unavailable: {type(e).__name__})"
    return {"img_id": img_id, "caption": caption}


def _content_to_text(content: Any, image_prefix: str = "") -> str:
    """契约 ContentPart[] 或字符串 → 文本（图像就地 caption 化，镜像官方 runner 文本模式）。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise HTTPException(400, "content must be a string or ContentPart[]")
    parts: List[str] = []
    for p in content:
        if not isinstance(p, dict):
            parts.append(str(p))
            continue
        t = p.get("type")
        if t == "text":
            parts.append(p.get("text", ""))
        elif t == "image_url":
            url = (p.get("image_url") or {}).get("url", "")
            if url:
                info = _caption_image(url)
                parts.append(f"{image_prefix}image:\nimage_id: {info['img_id']}\n"
                             f"image_caption: {info['caption']}")
        else:
            parts.append(p.get("text", "") if isinstance(p.get("text"), str) else "")
    return "\n".join(x for x in parts if x)


# ---- 记忆系统初始化（复用本地基线同款配置）----
def _build_memory():
    cfg = OmniMemoryConfig.create_default()
    cfg.set_unified_model(LLM_MODEL)
    cfg.embedding.model_name = "all-MiniLM-L6-v2"
    cfg.embedding.embedding_dim = 384
    return OmniMemAdapter(data_dir=str(DATA_DIR / "memory"), config=cfg)


_ADAPTER = _build_memory()
_LOCK = threading.Lock()

_seen_ids = set()
if SEEN_LOG.exists():
    with open(SEEN_LOG, encoding="utf-8") as f:
        for line in f:
            try:
                _seen_ids.add(json.loads(line)["request_id"])
            except Exception:
                pass

app = FastAPI(title="AML Multimodal Memory Service", version="0.1.0")


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


def _ts_to_iso(ts: Optional[Union[int, float, str]]) -> str:
    if ts is None:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        v = float(ts)
        if v > 1e12:  # 毫秒
            v /= 1000.0
        return datetime.fromtimestamp(v, tz=timezone.utc).isoformat(timespec="seconds")
    except Exception:
        return str(ts)


@app.get("/health")
async def health():
    return {"status": "ok", "model": LLM_MODEL, "service": "aml-multimodal-v0"}


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
    messages = payload.get("messages") or []
    user_id = str(payload.get("user_id") or "default_user")
    session_id = str(payload.get("session_id") or "default_session")
    ts_iso = _ts_to_iso(payload.get("timestamp"))

    # 幂等：重复 request_id 直接成功返回
    if request_id in _seen_ids:
        return {"status": "ok", "request_id": request_id, "duplicate": True}

    text_parts = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "user")
        t = _content_to_text(msg.get("content"))
        if t:
            text_parts.append(f"{role}: {t}")
    text = "\n".join(text_parts)

    if text:
        observation = {
            "text": text,
            "image": None,
            "timestamp": ts_iso,
            "dialogue_id": f"{session_id}:{request_id}",
        }
        with _LOCK:
            _ADAPTER.store(observation)  # 同步持久化（含摘要/实体抽取/索引）

    with open(SEEN_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"request_id": request_id, "ts": ts_iso}) + "\n")
    _seen_ids.add(request_id)
    return {"status": "ok", "request_id": request_id}


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
    if isinstance(query, list):
        qtext = _content_to_text(query, image_prefix="question's ")
    else:
        qtext = str(query)
    if not qtext:
        raise HTTPException(400, "empty query")

    top_k = payload.get("top_k") or 100
    with _LOCK:
        context = _ADAPTER.recall(qtext) or ""
    return {"data": [{"type": "text", "text": context}], "top_k": top_k}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
