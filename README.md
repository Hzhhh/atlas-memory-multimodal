# Atlas-Memory-Multimodal

Agent Memory Leaderboard (AML) Cycle 2 **多模态赛道**参赛方案：基于 [Omni-SimpleMem](https://github.com/aiming-lab/SimpleMem) 的多模态长期记忆系统，实现官方 `memory-api-v1.1` 契约的 Add / Search 服务。

## 架构（v0.2.1）

```
Add 请求 (messages, 含 base64 图像 ContentPart)
   │  文本拼接 + gpt-4o-mini 视觉 caption（图像文本化）
   ▼
Omni-SimpleMem 写入管线（同步持久化）
   │  语义摘要 / 实体抽取 / 混合索引（FAISS 向量 + BM25 + 时间/会话元数据）
   ▼
Search 请求 (query, 支持文本与图像)
   │  query 处理 + 复杂度自适应检索
   ▼
{"data": [{"content": "<检索上下文>"}]}
```

- **模型合规**：Add 与 Search 阶段全部 LLM 调用使用 `gpt-4o-mini`（官方要求，见服务环境变量 `AML_LLM_MODEL`）
- **嵌入**：all-MiniLM-L6-v2（本地，官方允许自由选择嵌入模型）
- **幂等**：按 `request_id` 去重（重启后仍有效，落盘于 `/data/request_ids.jsonl`）
- **鉴权**：`Authorization: Token/Bearer <key>` 或 `X-Api-Key: <key>`

## 部署

```bash
git clone https://github.com/Hzhhh/atlas-memory-multimodal.git
cd atlas-memory-multimodal
docker build -t aml-multimodal -f service/Dockerfile .

docker run -d --name aml-mm -p 8002:8000 \
  -v /root/aml-mm-data:/data \
  -e OPENAI_API_KEY=<你的key> \
  -e OPENAI_API_BASE=https://openrouter.ai/api/v1 \
  -e MEMORY_SYSTEM_KEY=<服务鉴权key> \
  aml-multimodal
```

端点：
- `GET  /health`
- `POST /v1/memory/add`（别名 `/add`）
- `POST /v1/memory/search`（别名 `/search`）

自测示例见 `docs/selftest.md`。

## 致谢与披露（开源合规）

本方案基于以下开源工作构建，遵循其原始许可证：

| 上游项目 | 许可证 | 论文 | 用途 |
|---|---|---|---|
| [Omni-SimpleMem](https://github.com/aiming-lab/SimpleMem)（aiming-lab） | Apache 2.0 | arXiv:2604.01007 | 记忆系统底座（orchestrator + Mem-Gallery 适配器） |
| [Mem-Gallery](https://github.com/YuanchenBei/Mem-Gallery) | 见上游 | arXiv:2601.03515 | 本地评测框架（开发期验证） |

**我们的改动**详见 [DISCLOSURE.md](docs/upstream_changes.md)：核心为新增 `service/aml_mm_service.py`（官方契约服务层：ContentPart 解析、base64 图像 caption 文本化、幂等、鉴权、token 记账），以及评测侧适配文件 `docs/omnimem_wrapper.py`。未修改上游核心记忆管线语义。

## 开发路线

- [x] v0.1.0：契约服务上线（Omni-SimpleMem 文本模式，本地 Mem-Gallery 子集 F1 验证）
- [x] v0.2.1：评测鲁棒性加固——快速同步 Add（<600ms@30并发，无 LLM 阻塞）+ 后台异步增强（OmniMem 管线/图像 caption）+ Search 限时降级；压力自测通过
- [ ] v0.2+：检索优化（强嵌入模型、时间线索引、知识冲突处理）——详见技术报告
