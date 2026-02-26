# 2. RAG架构详解 (Architectures)

本模块详细介绍23种RAG架构，包括经典9种和补充的14种。

> ✅ = 已完成 | ⏳ = 编写中

## 📋 目录

### 基础型架构
| 架构 | 文件 | 描述 | 状态 |
|------|------|------|------|
| 标准RAG | [standard-rag.md](./architectures/standard-rag.md) | 单次检索，拼接上下文生成 | ✅ |
| 对话式RAG | [conversational-rag.md](./architectures/conversational-rag.md) | 维护对话历史，动态检索 | ✅ |

### 交互型架构
| 架构 | 文件 | 描述 | 状态 |
|------|------|------|------|
| Corrective RAG | [corrective-rag.md](./architectures/corrective-rag.md) | 检索后验证、重写或补充 | ⏳ |
| Adaptive RAG | [adaptive-rag.md](./architectures/adaptive-rag.md) | 根据查询复杂度选择策略 | ⏳ |

### 自反思架构
| 架构 | 文件 | 描述 | 状态 |
|------|------|------|------|
| Self-RAG | [self-rag.md](./architectures/self-rag.md) | 生成过程中自我反思与检索 | ✅ |

### 融合型架构
| 架构 | 文件 | 描述 | 状态 |
|------|------|------|------|
| Fusion RAG | [fusion-rag.md](./architectures/fusion-rag.md) | 多路检索结果融合排序 | ⏳ |
| HyDE | [hyde.md](./architectures/hyde.md) | 先生成假设文档再检索 | ⏳ |

### 智能体型架构
| 架构 | 文件 | 描述 | 状态 |
|------|------|------|------|
| Agentic RAG | [agentic-rag.md](./architectures/agentic-rag.md) | 结合Agent规划与工具调用 | ✅ |

### 图结构架构
| 架构 | 文件 | 描述 | 状态 |
|------|------|------|------|
| GraphRAG | [graph-rag.md](./architectures/graph-rag.md) | 利用知识图谱增强检索 | ✅ |

### 迭代检索架构
| 架构 | 文件 | 描述 | 状态 |
|------|------|------|------|
| REPLUG | [replug.md](./architectures/replug.md) | 检索器与LLM解耦，加权输入 | ⏳ |
| Iterative RAG | [iterative-rag.md](./architectures/iterative-rag.md) | 多跳检索与生成交替 | ⏳ |

### 其他高级架构
| 架构 | 文件 | 描述 | 状态 |
|------|------|------|------|
| In-Context RALM | [in-context-ralm.md](./architectures/in-context-ralm.md) | 不微调，仅靠上下文学习 | ⏳ |
| FLARE | [flare.md](./architectures/flare.md) | 生成中主动触发检索 | ⏳ |
| Multimodal RAG | [multimodal-rag.md](./architectures/multimodal-rag.md) | 图文音视频联合检索 | ⏳ |
| RA-CM3 | [ra-cm3.md](./architectures/ra-cm3.md) | 检索图文对指导图像生成 | ⏳ |
| Self-Ask + RAG | [self-ask-rag.md](./architectures/self-ask-rag.md) | 自问自答，逐层检索 | ⏳ |
| SQL-RAG | [sql-rag.md](./architectures/sql-rag.md) | NL转SQL查询数据库 | ⏳ |
| Table RAG | [table-rag.md](./architectures/table-rag.md) | 针对表格的检索推理 | ⏳ |
| MapReduce RAG | [mapreduce-rag.md](./architectures/mapreduce-rag.md) | 分而治之，合并摘要 | ⏳ |
| DRAGIN | [dragin.md](./architectures/dragin.md) | 根据注意力动态检索 | ⏳ |
| Self-Mem | [self-mem.md](./architectures/self-mem.md) | 结合外部检索与内部记忆 | ⏳ |
| RAPTOR | [raptor.md](./architectures/raptor.md) | 构建文档摘要树，分层检索 | ⏳ |

## 🎯 学习路径

```
基础 ───────────────────────────────────────────────────▶ 高级

标准RAG ─▶ 对话式RAG ─▶ Fusion RAG ─▶ HyDE ─▶ 自适应RAG
                            │
                            ▼
                    Agentic RAG ─▶ Self-RAG ─▶ GraphRAG
```

## 📖 架构对比

| 架构 | 检索次数 | 复杂度 | 适用场景 |
|------|----------|--------|----------|
| 标准RAG | 1次 | ⭐ | 简单问答 |
| 对话式RAG | 多轮 | ⭐⭐ | 对话系统 |
| Agentic RAG | 动态 | ⭐⭐⭐⭐ | 复杂推理 |
| GraphRAG | 多跳 | ⭐⭐⭐ | 关系密集场景 |

## 🔬 实战

每个架构都配有：
- 论文/来源链接
- 核心原理图解
- 代码示例
- Notebook实战

## ⏭️ 下一步

进入 [3-advanced/README.md](../3-advanced/README.md) 学习高级专题。

---

> ⚠️ **注意**: 
> - ✅ = 已完成，可直接阅读
> - ⏳ = 正在编写中，暂不可访问
