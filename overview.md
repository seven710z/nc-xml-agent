# 交付总览 —— 数控 XML 界面生成与自动部署 AI Agent 面试 Demo

## 一句话总结

基于实习项目描述，重建了【数控 XML 界面生成与自动部署 AI Agent】的最小可运行 Demo + 全套面试讲解材料，聚焦本人独立负责的 6 大模块，全程贯穿「底层原理 → 技术选型 → 踩坑 → trade-off → 工程落地」的逻辑链。

## 交付物清单

| # | 交付物 | 路径 | 说明 |
|---|---|---|---|
| 1 | 可运行工程 | `nc-agent-demo/` | FastAPI + LangGraph + LCEL + MCP + Chroma RAG + Qwen2.5/LoRA + XML 生成校验 |
| 2 | 底层讲解稿 | `docs/01_底层讲解稿.md` | 3分钟开场 + 11 层底层串讲（大模型→LoRA→结构化→RAG→Tool Calling→MCP→LangGraph→LCEL→asyncio→FastAPI→链路串联） |
| 3 | 面试深挖题库 | `docs/02_面试深挖题库.md` | 20 道高频追问（底层原理/选型/踩坑/trade-off/优化）+ 陷阱反问 |
| 4 | 项目亮点提炼 | `docs/03_项目亮点提炼.md` | 3 条核心亮点 + 简历版 + 30秒开场版 |
| 5 | 依赖与启动 | `nc-agent-demo/requirements.txt` + `README.md` + `.env.example` | 含 mock 模式降级说明 |

## 核心设计决策

1. **mock 模式竖切片**：默认 `APP_MODE=mock`，无需 API key/GPU/向量库即可跑通完整链路（LLM 用确定性规则、RAG 用字符 n-gram 哈希 embedding 降级），面试现场零依赖演示；real 模式走真模型 + Chroma + MCP。
2. **可插拔降级**：LLM 客户端、Embedder、MCP 客户端均设计成「对外签名一致、内部可降级」，体现工程健壮性（C4）。
3. **三层 XML 校验**：良构性 → schema → 业务绑定规则，把「不确定输出」关进「确定性校验」框架。
4. **结构敏感 LoRA**：loss 按 token 类型重加权，是项目最有区分度的技术点。

## 已验证

- ✅ `scripts/demo_run.py` 完整链路跑通（6 步：解析→检索→生成→校验→确认→部署），XML 生成正确、RAG 正确召回 3 类知识、input 缺 min/max 的 warning 正确触发。
- ✅ FastAPI `/health`、`/generate` 接口返回 200，端到端 17ms。

## 使用方式

```bash
cd nc-agent-demo
pip install -r requirements.txt
cp .env.example .env
python scripts/demo_run.py          # 命令行演示
# 或
python -m app.api.main              # HTTP 服务
```

## 后续建议（面试前可再做）

1. 准备一份「真实 GPU 训练 LoRA 的 loss 曲线/效果对比」截图（若有），增强结构敏感 LoRA 的说服力。
2. 把讲解稿通读 2~3 遍，重点记「每一层技术解决上一层什么缺陷」这条因果链。
3. 若面试官现场要求「真 RAG」，可临时 `pip install chromadb sentence-transformers` 切 real 模式演示。
