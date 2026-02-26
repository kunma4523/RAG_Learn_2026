# 重排序

重排序(Reranking)使用更精确的模型对初检结果进行二次排序。

## 1. Cross-Encoder

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# 重新打分
pairs = [(query, doc) for doc in docs]
scores = model.predict(pairs)
```

## 2. 使用场景

- 初检Top100 → 重排Top10
- 候选集较小时使用
- 平衡效果与延迟
