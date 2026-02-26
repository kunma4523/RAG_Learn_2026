# 检索指标

评估检索质量的指标。

## 1. 常用指标

- Recall@K: 前K个结果中相关文档的比例
- Precision@K: 前K个结果中相关文档的精确率
- MRR: 第一个相关文档排名的倒数
- NDCG: 考虑排名的归一化折扣增益

## 2. 计算

```python
def recall_at_k(retrieved, relevant, k):
    retrieved_k = set(retrieved[:k])
    relevant_set = set(relevant)
    return len(retrieved_k & relevant_set) / len(relevant_set)

def mrr(retrieved, relevant):
    for i, doc in enumerate(retrieved, 1):
        if doc in relevant:
            return 1.0 / i
    return 0.0
```
