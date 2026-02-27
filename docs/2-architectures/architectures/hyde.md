# HyDE (Hypothetical Document Embeddings)

**架构名称**: HyDE  
**类型**: 假设型  
**难度**: ⭐⭐  

## 1. 核心思想

HyDE的核心思想是：先用LLM生成一个"假设文档"，然后用这个假设文档去检索真实文档。

## 2. 工作流程

```
Query: "什么是RAG?"
    │
    ▼
┌──────────────────────────────────────┐
│  Step 1: 生成假设文档               │
│  LLM生成一个"理想答案"文档           │
│  "RAG是检索增强生成..."             │
└──────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────┐
│  Step 2: 用假设文档检索              │
│  向量化假设文档，检索相似真实文档     │
└──────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────┐
│  Step 3: 返回真实文档               │
│  返回top-k相关文档                   │
└──────────────────────────────────────┘
```

## 3. 代码示例

```python
# 运行测试脚本
python tests/rag_architectures/07_hyde.py
```

或查看完整代码:
- 测试代码: `tests/rag_architectures/07_hyde.py`

## 4. 适用场景

```python
from langchain.chains importhyde

chain = hyde.HypotheticalDocumentEmbedder.from_llm(
    llm=llm,
    base_embeddings=embeddings,
    prompt_key="web_search"  # or "chat"
)
```

## 4. 适用场景

- 搜索词与文档风格差异大
- 短查询
- 开放域问答
