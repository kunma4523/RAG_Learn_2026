# 检索模块

检索是RAG系统的第一步，负责从知识库中找到与查询最相关的文档。

## 1. 检索流程

```
┌─────────────────────────────────────────────────────────────┐
│                      检索流程                                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐    ┌─────────────┐    ┌──────────────┐     │
│  │  Query   │───▶│ Query Processing│───▶│ Vector/Match│     │
│  │ "什么是RAG"│    │ 分词/向量化   │    │  相似度计算  │     │
│  └──────────┘    └─────────────┘    └──────┬───────┘     │
│                                            │              │
│                                            ▼              │
│                                   ┌──────────────┐        │
│                                   │  Top-K Docs  │        │
│                                   │  [doc1,doc2] │        │
│                                   └──────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

## 2. 检索方法分类

### 2.1 稀疏检索

| 方法 | 原理 | 特点 |
|------|------|------|
| TF-IDF | 词频-逆文档频率 | 简单快速 |
| BM25 | 改进的词频模型 | 效果好，通用 |

### 2.2 密集检索

| 方法 | 原理 | 特点 |
|------|------|------|
| DPR | 双编码器 | 语义匹配 |
| Contriever | 对比学习 | 无监督预训练 |
| BGE | 中英文语义 | 效果好 |

### 2.3 混合检索

结合稀疏和密集方法，取长补短。

## 3. 实现示例

### 3.1 基础密集检索

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SimpleRetriever:
    def __init__(self, model_name="BAAI/bge-base-zh-v1.5"):
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = None
    
    def index(self, documents):
        self.documents = documents
        self.embeddings = self.model.encode(documents)
    
    def retrieve(self, query, top_k=5):
        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # 排序
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [
            {"text": self.documents[i], "score": float(similarities[i])}
            for i in top_indices
        ]

# 使用
retriever = SimpleRetriever()
retriever.index(["文档1内容", "文档2内容", "文档3内容"])
results = retriever.retrieve("查询内容")
```

### 3.2 带元数据过滤

```python
class FilterableRetriever:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.metadatas = []
    
    def index(self, documents, metadatas):
        self.documents = documents
        self.metadatas = metadatas
        self.embeddings = self.model.encode(documents)
    
    def retrieve(self, query, top_k=5, filter_fn=None):
        # 先过滤
        if filter_fn:
            indices = [i for i, m in enumerate(self.metadatas) 
                      if filter_fn(m)]
        else:
            indices = list(range(len(self.documents)))
        
        # 向量检索
        query_embedding = self.model.encode([query])
        doc_embeddings = np.array([self.embeddings[i] for i in indices])
        similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
        
        # 排序返回
        sorted_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [
            {
                "text": self.documents[indices[i]], 
                "score": float(similarities[i]),
                "metadata": self.metadatas[indices[i]]
            }
            for i in sorted_indices
        ]
```

## 4. 检索优化技巧

| 技巧 | 说明 |
|------|------|
| 查询扩展 | 用同义词扩展查询 |
| 查询重写 | 用LLM改写查询 |
| 混合检索 | 结合BM25和向量检索 |
| 重排序 | 用Cross-Encoder二次排序 |

## 下一步

- [增强模块](./4-augmentation.md) - 学习如何增强上下文
