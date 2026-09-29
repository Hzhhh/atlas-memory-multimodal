"""OmniMem (Omni-SimpleMem) 桥接包装器。

把 OmniSimpleMem 仓库里的 OmniMemAdapter 注册进 Mem-Gallery 官方评测框架：
run_bench.py 中 `eval(memory_name)(MemoryConfig(config))` 会按名字找到本文件的 OmniMem 类。

用法:
    python run_bench.py --llm_name gpt-4o-mini --memory_name OmniMem \
        --data_name <场景名> --save_results

环境变量:
    OMNISIMPLEMEM_ROOT  SimpleMem/OmniSimpleMem 仓库路径（默认自动探测）
    OMNIMEM_DATA_DIR    记忆持久化目录（默认 ./omni_memory_eval_data）
    OMNIMEM_LLM_MODEL   OmniMem 内部用的 LLM（冒烟用 gpt-4o-mini 省钱；忠实复现用 gpt-4o）
    OPENAI_API_KEY      必须
"""
import os
import sys
import itertools

# 项目根：环境变量优先，否则从本文件位置推导（Mem-Gallery/benchmark/run/ 上三级）
_PROJECT_ROOT = os.environ.get(
    "AML_PROJECT_ROOT",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)
OMNI_ROOT = os.environ.get("OMNISIMPLEMEM_ROOT", os.path.join(_PROJECT_ROOT, "SimpleMem", "OmniSimpleMem"))
if OMNI_ROOT not in sys.path:
    sys.path.insert(0, OMNI_ROOT)

from benchmarks.memgallery.adapter import OmniMemAdapter  # noqa: E402
from omni_memory import OmniMemoryConfig as _OmniCfg  # noqa: E402


# ---- 全局 token 记账：覆盖 OmniMem 内部 LLM 与评测答题的所有 chat 调用 ----
import json as _json  # noqa: E402
from datetime import datetime as _dt  # noqa: E402

USAGE_LOG = os.path.join(_PROJECT_ROOT, "tasks", "token_usage.jsonl")


def _install_usage_accounting() -> None:
    """monkey-patch openai Completions.create，把每次调用的 usage 追加到 JSONL。"""
    try:
        from openai.resources.chat.completions import Completions
        _orig = Completions.create

        def _create(self, *a, **kw):
            resp = _orig(self, *a, **kw)
            try:
                u = getattr(resp, "usage", None)
                if u is not None:
                    with open(USAGE_LOG, "a", encoding="utf-8") as f:
                        f.write(_json.dumps({
                            "ts": _dt.now().isoformat(timespec="seconds"),
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
# ---- 记账结束 ----


def load_dotenv_local(path: str = None) -> None:
    """极简 .env 加载（KEY=VALUE 每行一条），不覆盖已有环境变量。"""
    path = path or os.path.join(_PROJECT_ROOT, ".env")
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


class _RecallOpStub:
    """memengine 检索指标钩子：--eval_retrieval_metrics 时框架会读 last_retrieved_ids。"""

    def __init__(self):
        self.last_retrieved_ids = []


_reset_counter = itertools.count(1)


class OmniMem:
    """memengine 兼容外壳：is_multimodal=False（caption 并入文本，论文设置）。"""

    def __init__(self, config):
        args = getattr(config, "args", None)
        omni_attr = getattr(args, "omni", None) if args is not None else None
        if isinstance(omni_attr, dict):
            oc = dict(omni_attr)
        elif omni_attr is not None:
            oc = dict(vars(omni_attr))
        else:
            oc = {}

        self._base_data_dir = oc.get(
            "data_dir", os.environ.get("OMNIMEM_DATA_DIR", "./omni_memory_eval_data")
        )
        llm_model = oc.get("llm_model", os.environ.get("OMNIMEM_LLM_MODEL", "gpt-4o-mini"))
        config_yaml = oc.get("config_yaml", os.path.join(OMNI_ROOT, "configs", "memgallery_config.yaml"))

        # 优先用论文官方 yaml 构建，再覆盖 LLM 型号（控成本）
        if os.path.exists(config_yaml):
            import yaml
            with open(config_yaml, "r", encoding="utf-8") as f:
                y = yaml.safe_load(f) or {}
            omni_cfg = _OmniCfg.from_dict(y)
        else:
            omni_cfg = _OmniCfg.create_default()
        omni_cfg.set_unified_model(llm_model)
        # 本地文本嵌入，无 API 依赖
        omni_cfg.embedding.model_name = "all-MiniLM-L6-v2"
        omni_cfg.embedding.embedding_dim = 384

        self._omni_cfg = omni_cfg
        self._adapter = None
        self.recall_op = _RecallOpStub()

    # ---- memengine ExplicitMemory 接口 ----
    def reset(self):
        """每次重置换全新隔离目录：保证多场景评测互不串扰（含历史遗留数据）。"""
        scene_dir = os.path.join(self._base_data_dir, f"scene_{next(_reset_counter):03d}")
        os.makedirs(scene_dir, exist_ok=True)
        self._adapter = OmniMemAdapter(data_dir=scene_dir, config=self._omni_cfg)

    def store(self, observation):
        self._adapter.store(observation)

    def recall(self, query):
        result = self._adapter.recall(query)
        return result


# run_bench.py 的 DialogueAgentMemoryConfig 用
DEFAULT_OMNIMEM = {
    "global_config": {"usable_gpu": ""},   # MemoryConfig 构造必读
    "is_multimodal": False,                # 文本模式：图像以 caption 形式并入文本
    "omni": {},                            # 具体项由环境变量提供，见文件头
}
