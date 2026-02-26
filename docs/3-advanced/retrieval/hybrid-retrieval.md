# 混合检索

混合检索结合稀疏检索和密集检索的优点。

## 1. 为什么需要混合检索

- 稀疏检索：对精确匹配好
- 密集检索：对语义匹配好
- 混合：取长补短

## 2. 实现

```python
# BM25 + Dense 混合检索
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# 两种检索
bm25_scores = ...
dense_scores = ...

# 融合 (RRF)
k = 60
combined_scores = {}
for i, (bm25, dense) in enumerate(zip(bm25_scores, dense_scores)):
    combined_scores[i] = bm25/(k+i+1) + dense/(k+i+1)
```

## 3. 最佳实践

- 调节alpha权重
- 使用RRF融合
- 先粗筛后精排
