# Fusion RAG

**架构名称**: Fusion RAG  
**类型**: 融合型  
**难度**: ⭐⭐  

## 1. 核心思想

融合多种检索方法的结果：
- BM25 (稀疏检索)
- 向量检索 (密集检索)
- 其他自定义检索器

## 2. 融合方法

### Reciprocal Rank Fusion (RRF)

```python
def reciprocal_rank_fusion(results_list, k=60):
    """RRF算法"""
    scores = {}
    
    for results in results_list:
        for rank, doc in enumerate(results):
            doc_id = doc["id"]
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    
    # 排序
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs
```

## 3. 完整实现

```python
# 多种检索器
bm25_retriever = BM25Retriever.from_documents(docs)
dense_retriever = VectorStoreIndex.from_documents(docs).as_retriever()

# 融合检索
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.fusion import FusionRetriever

fusion_retriever = FusionRetriever(
    retrievers=[bm25_retriever, dense_retriever],
    fusion_mode="reciprocal_rank_fusion"
)
```
