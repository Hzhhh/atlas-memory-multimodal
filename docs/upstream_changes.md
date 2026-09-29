# 上游仓库改动披露清单

> AML 官方要求：基于已有仓库改造必须披露原作者 + 技术报告 + 全部改动。
> 本文件随开发持续更新，提交评测申请时整理进技术报告。

## 1. SimpleMem / OmniSimpleMem（底座）
- 原仓库：https://github.com/aiming-lab/SimpleMem（Apache 2.0）
- 论文：Omni-SimpleMem (arXiv:2604.01007)；SimpleMem (arXiv:2601.02553)
- 作者：Liu, Jiaqi 等（aiming-lab）
- 用途：记忆系统底座（OmniMemoryOrchestrator + Mem-Gallery 适配器）
- 我们的改动：见下方第 3 节

## 2. Mem-Gallery 评测框架
- 原仓库：https://github.com/YuanchenBei/Mem-Gallery
- 论文：Mem-Gallery (arXiv:2601.03515, ACL 2026)；数据 HF: Ethan-Bei/Mem-Gallery (MIT)
- 用途：本地评测 harness（run_bench.py + memengine）

### 对 Mem-Gallery 框架的改动
| 文件 | 改动 | 原因 |
|---|---|---|
| `benchmark/run/run_bench.py` | 新增 OmniMem 注册（import + elif 分支）；gpt-4o-mini 分支的 API 端点从 openrouter 硬编码改为读环境变量 OPENAI_API_KEY/OPENAI_BASE_URL/OPENAI_MODEL_NAME | 注册我们的参赛记忆系统；用官方 OpenAI 端点 |
| `benchmark/run/omnimem_wrapper.py` | **新增文件**：OmniMem 桥接包装器（memengine 接口适配 + .env 加载 + recall_op 指标钩子） | 参赛系统集成 |
| `benchmark/memengine/function/MultiModalRetrieval.py` | MMEmbedEncoder 导入改为 try/except 容错 + 使用处兜底报错 | 上游 bug：该类已在 MultiModalEncoder.py 中移除但导入残留，阻断 import |
| `benchmark/memengine/evaluate/evaluation.py` | --memory_name choices 白名单加入 OmniMem | 注册参赛系统 |
| `benchmark/default_config/DefaultGlobalConfig.py` | DEFAULT_BACKBONE_PATH 从 '' 改为 'gpt2' | 上游 bug：空路径导致 LMTruncation 的 AutoTokenizer 报 "Repo id ''" 错误；gpt2 为 GPT 系标准 tokenizer，语义不变 |
| `benchmark/run/run_bench.py` | --memory_name choices 白名单加入 OmniMem | 同上注册 |

**评测语义未改动**：数据加载、QA 循环、prompt、F1/LLM-judge 计分逻辑保持原样。

## 3. 对 OmniSimpleMem 的改动
（基线复现阶段暂无代码改动；后续优化逐条记录在此）

| 文件 | 改动 | 动机/效果 |
|---|---|---|

## 4. ATM-Bench 框架
- 原仓库：https://github.com/JingbiaoMei/ATM-Bench（MIT，数据 CC-BY-NC 4.0）
- 论文：arXiv:2603.01990
- 改动：暂无
