# Adaptive RAG

**架构名称**: Adaptive RAG  
**类型**: 自适应型  
**难度**: ⭐⭐⭐  

## 1. 核心思想

Adaptive RAG让模型自己决定何时检索、检索什么。

## 2. 核心逻辑

```python
def adaptive_rag(query):
    # 模型决定是否需要检索
    if should_retrieve(query):
        # 检索
        docs = retrieve(query)
        
        # 决定使用多少文档
        relevant_docs = filter_relevant(docs)
        
        # 生成
        return generate(query, relevant_docs)
    else:
        # 直接生成
        return generate_without_retrieval(query)
```

## 3. 实现

```python
from langchain.agents import AgentExecutor, create_self_ask_with_search_agent

# 使用ReAct风格的Agent
agent = create_self_ask_with_search_agent(llm, tools)
```
