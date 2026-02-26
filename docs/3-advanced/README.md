# 3. 高级专题 (Advanced Topics)

本模块深入探讨RAG系统的高级优化技术，包括检索优化、生成优化、评估体系和部署实践。

## 📋 目录

> ⚠️ 带 ⏳ 标记的文档正在编写中

### 3.1 检索优化
- [混合检索](./retrieval/hybrid-retrieval.md) - 稀疏+密集检索融合 ⏳
- [重排序](./retrieval/reranking.md) - Cross-Encoder重排序 ⏳
- [查询改写](./retrieval/query-rewriting.md) - 查询扩展与改写 ⏳
- [索引优化](./retrieval/index-optimization.md) - 索引策略与压缩 ⏳

### 3.2 生成优化
- [提示压缩](./generation/prompt-compression.md) - 提示词压缩技术 ⏳
- [长上下文](./generation/long-context.md) - 长上下文窗口利用 ⏳
- [解码策略](./generation/decoding-strategies.md) - 解码策略优化 ⏳

### 3.3 评估体系
- [检索指标](./evaluation/retrieval-metrics.md) - Recall@k, MRR, NDCG ⏳
- [生成指标](./evaluation/generation-metrics.md) - BLEU, ROUGE, BERTScore ⏳
- [端到端评估](./evaluation/e2e-evaluation.md) - 正确率、忠实度 ⏳
- [RAG评估工具](./evaluation/rag-evaluation.md) - RAGAS、LLM-as-Judge ⏳

### 3.4 效率与部署
- [检索缓存](./deployment/caching.md) - 检索结果缓存策略 ⏳
- [模型量化](./deployment/quantization.md) - 模型压缩与量化 ⏳
- [服务化部署](./deployment/serving.md) - API服务搭建 ⏳
- [负载均衡](./deployment/load-balancing.md) - 多实例部署 ⏳

### 3.5 隐私与安全
- [隐私泄露](./security/privacy-leakage.md) - RAG中的隐私问题 ⏳
- [对抗攻击](./security/adversarial-attacks.md) - 对抗样本攻击 ⏳
- [防御策略](./security/defense-strategies.md) - 安全防护措施 ⏳

## 🎯 学习目标

1. 掌握检索和生成的高级优化技术
2. 建立完整的RAG评估体系
3. 了解生产级部署的最佳实践

## 📊 评估框架

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG Evaluation Pipeline                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐                                           │
│  │   Input    │                                           │
│  │  (Query +  │                                           │
│  │  Context)  │                                           │
│  └──────┬──────┘                                           │
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │ Retrieval  │───▶│   LLM       │───▶│  Evaluation │     │
│  │  Metrics   │    │   Output    │    │   Metrics   │     │
│  │            │    │             │    │             │     │
│  │ • Recall@k│    │ • Answer    │    │ • Faithful. │     │
│  │ • MRR     │    │ • Context   │    │ • Answer    │     │
│  │ • NDCG    │    │ • Sources   │    │   Relevan.  │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 🛠️ 工具推荐

| 类别 | 工具 |
|------|------|
| 评估 | RAGAS, DeepEval, LangChain Evaluation |
| 监控 | LangSmith, Arize AI, Helicone |
| 部署 | vLLM, Text Generation Inference, Ray Serve |

## ⏭️ 下一步

进入 [4-future/README.md](../4-future/README.md) 了解最新前沿趋势。

---

> ⚠️ **注意**: 带 ⏳ 标记的文档正在编写中，敬请期待！
