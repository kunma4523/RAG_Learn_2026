# Corrective RAG

**架构名称**: Corrective RAG (CRAG)  
**类型**: 自校正型  
**难度**: ⭐⭐⭐  

## 1. 核心思想

CRAG在检索后增加自校正机制：
1. 评估检索结果质量
2. 过滤低质量文档
3. 必要时重写查询或检索

## 2. 工作流程

```
┌──────────┐    ┌─────────────┐    ┌──────────────┐
│   Query  │───▶│  Retrieval  │───▶│  Evaluation │
│          │    │             │    │ (质量评估)   │
└──────────┘    └─────────────┘    └──────┬───────┘
                                            │
                         ┌─────────────────┬─────────────────┐
                         ▼                 ▼                 ▼
                    高质量             中等质量            低质量
                     │                   │                   │
                     ▼                   ▼                   ▼
                 生成答案            补充检索          查询重写+重检索
```

## 3. 实现要点

```python
# 评估检索质量
def evaluate_retrieval(query, docs):
    # 检查文档相关性
    # 检查覆盖率
    # 返回质量分数
    pass

# 决策逻辑
if quality > 0.8:
    return docs  # 直接使用
elif quality > 0.5:
    return retrieve_more()  # 补充检索
else:
    return rewrite_and_retrieve()  # 重写检索
```
