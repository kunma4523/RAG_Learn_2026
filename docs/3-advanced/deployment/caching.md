# 检索缓存

缓存检索结果，提高响应速度。

## 1. 缓存策略

- 完全匹配缓存
- 近似匹配缓存
- LRU缓存

## 2. 实现

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_retrieve(query):
    return retrieve(query)
```
