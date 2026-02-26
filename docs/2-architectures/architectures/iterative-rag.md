# Iterative RAG

**架构名称**: Iterative RAG  
**类型**: 迭代检索型  
**难度**: ⭐⭐⭐  

## 1. 核心思想

迭代检索与生成：
1. 检索 → 生成 → 评估 → 必要时再检索
2. 适合多跳问答
3. 逐步深入理解问题

## 2. 工作流程

```
Query: "谁写的《百年孤独》？"
    │
    ├── 检索: "《百年孤独》"
    ├── 生成: "《百年孤独》是加西亚·马尔克斯写的"
    └── 评估: ✓ 回答正确

Query: "他还有什么代表作？"
    │
    ├── 检索: "加西亚·马尔克斯 代表作"
    ├── 生成: "还有《霍乱时期的爱情》等"
    └── 评估: ✓ 回答正确
```

## 3. 实现

```python
def iterative_rag(query, max_iterations=3):
    for i in range(max_iterations):
        # 检索
        docs = retriever.retrieve(query)
        
        # 生成
        answer = generator.generate(query, docs)
        
        # 检查是否需要继续
        if check_completeness(answer):
            return answer
        
        # 提取下一个查询
        query = extract_next_query(answer)
    
    return final_answer
```
