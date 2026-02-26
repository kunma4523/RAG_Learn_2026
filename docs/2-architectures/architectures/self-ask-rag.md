# Self-Ask + RAG

**架构名称**: Self-Ask + RAG  
**类型**: 子问题分解型  
**难度**: ⭐⭐⭐  

## 1. 核心思想

Self-Ask让模型自己分解问题：
1. 分析主问题
2. 分解为子问题
3. 逐个检索回答子问题
4. 汇总生成最终答案

## 2. 示例

```
Q: "谁写了《百年孤独》？它获得了什么奖项？"

Self-Ask分析:
- 子问题1: 谁写了《百年孤独》?
- 子问题2: 《百年孤独》获得了什么奖项?

检索回答各子问题后，汇总答案
```

## 3. 实现

```python
from langchain import self_ask_with_search

agent = self_ask_with_search.create_self_ask_with_search_agent(llm, search_tool)
```
