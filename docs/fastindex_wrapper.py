"""FastIndex 桥接包装器（线上 v0.2.9 服务同构检索 → Mem-Gallery 评测）。

与 omnimem_wrapper.py 平行：注册 FastIndexMem 类，store/recall 走与生产
aml_mm_service.FastIndex 完全相同的代码路径（清洁文本 → MiniLM 向量 → BM25 混合
→ 0.50 阈值过滤），保证 L2 分数与线上行为可比。

用法:
    python run_bench.py --llm_name gpt-4o-mini --memory_name FastIndexMem \
        --data_name <场景名> --save_results

环境变量:
    FASTINDEX_DATA_DIR  每场景隔离数据目录（默认 ./fastindex_eval_data）
"""
import os
import sys
import uuid
from pathlib import Path

_PROJECT_ROOT = os.environ.get(
    "AML_PROJECT_ROOT",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)

# 复用本机 MiniLM 缓存；DATA_DIR 仅是模块级默认，实际按场景重定向
_hf = Path.home() / ".cache" / "huggingface"
if _hf.exists():
    os.environ.setdefault("HF_HOME", str(_hf))
os.environ.setdefault("DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "fastindex_eval_data"))

sys.path.insert(0, os.path.join(_PROJECT_ROOT, "service"))
import aml_mm_service as svc  # noqa: E402  FastIndex/打分逻辑与线上同文件


_reset_counter = None  # 保留字段避免外部引用报错；场景目录已改用 uuid


class _RecallOpStub:
    """检索指标钩子：--eval_retrieval_metrics 时框架读 last_retrieved_ids。"""

    def __init__(self):
        self.last_retrieved_ids = []


class FastIndexMem:
    """memengine 兼容外壳：is_multimodal=False（caption 已由 run_bench 并入 text）。"""

    def __init__(self, config):
        self.base_dir = os.environ.get(
            "FASTINDEX_DATA_DIR",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "fastindex_eval_data"),
        )
        self.recall_op = _RecallOpStub()
        self._fast = None
        self._user = "scene_user"

    def reset(self):
        """每场景全新隔离目录：uuid 命名，进程内/跨进程都不可能与旧场景撞车。"""
        scene_dir = os.path.join(self.base_dir, f"scene_{uuid.uuid4().hex[:12]}")
        os.makedirs(scene_dir, exist_ok=True)
        # FastIndex 的日志路径是模块级全局，按场景重定向
        svc.TURNS_LOG = Path(scene_dir) / "turns.jsonl"
        svc.SEEN_LOG = Path(scene_dir) / "request_ids.jsonl"
        svc.SEARCH_LOG = Path(scene_dir) / "searches.jsonl"
        self._fast = svc.FastIndex(Path(scene_dir))

    def store(self, message_dict):
        """与生产 Add 同构：'[{ts}] {行}' 序列化后走 add_turn（切分为 msg/caption 单元）。"""
        if isinstance(message_dict, dict):
            text_in = message_dict.get("text", "")
            ts = str(message_dict.get("timestamp") or "")
        else:
            text_in, ts = str(message_dict), ""
        if not text_in:
            return
        text = "\n".join(f"[{ts}] {ln}" for ln in text_in.split("\n") if ln.strip())
        self._fast.add_turn({"request_id": uuid.uuid4().hex, "user_id": self._user,
                             "session_id": ts or "s0", "ts": ts, "text": text})

    def recall(self, query):
        if getattr(self._fast, "_pending", None):
            self._fast.drain(timeout=1200)  # benchmark 前排空编码，保证 dense 通道完整
        """与生产 Search 同构：k=100、0.50 阈值；空结果→空上下文（弃答信号同线上）。"""
        res = self._fast.search(str(query), user_id=self._user, k=100)
        self.recall_op.last_retrieved_ids = [r["id"] for r in res[:10]]
        if not res:
            return ""
        return "\n\n".join(f"[{i}] {r['content']}" for i, r in enumerate(res, 1))


# run_bench.py 的 DialogueAgentMemoryConfig 用
DEFAULT_FASTINDEX = {
    "global_config": {"usable_gpu": ""},  # MemoryConfig 构造必读
    "is_multimodal": False,               # 文本模式：caption 已并入文本
}
