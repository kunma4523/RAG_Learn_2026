# 向量数据库

向量数据库用于存储和检索高维向量，是RAG系统密集检索的核心组件。

## 1. 常用向量数据库

```
┌─────────────────────────────────────────────────────────────┐
│                   向量数据库对比                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  数据库          类型      优点                    适用场景     │
│  ───────────────────────────────────────────────────────   │
│  Faiss           开源      速度快，GPU加速        研究/小规模 │
│  Milvus          云/开源   功能丰富，分布式       生产环境     │
│  Pinecone        云服务     托管简单，扩展性好      云部署      │
│  Weaviate        开源/云   GraphQL，原生支持RAG   中小规模    │
│  Qdrant          开源/云   高性能，易用           中小规模    │
│  Chromadb        开源      轻量，简单              快速原型    │
│  Milvus          开源      功能完整               大规模生产  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 2. Faiss

### 特点
- Facebook开源
- 纯C++实现，支持GPU加速
- 多种索引类型
- 适合研究和小规模应用

### 使用

```python
import faiss
import numpy as np

# 假设有10000个512维向量
d = 512
n = 10000
vectors = np.random.random((n, d)).astype('float32')

# 1. 暴力索引 (Exact)
index_exact = faiss.IndexFlatL2(d)
index_exact.add(vectors)

# 搜索
query = np.random.random((1, d)).astype('float32')
distances, indices = index_exact.search(query, k=5)

# 2. IVF索引 (加速)
nlist = 100  # 聚类中心数
quantizer = faiss.IndexFlatL2(d)
index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist)
index_ivf.train(vectors)
index_ivf.add(vectors)

# 3. HNSW索引 (高速)
index_hnsw = faiss.IndexHNSWFlat(d, 32)  # 32是每个节点的连接数
index_hnsw.add(vectors)
```

## 3. Pinecone

### 特点
- 全托管云服务
- 简单易用
- 自动扩展

### 使用

```python
from pinecone import Pinecone

# 初始化
pc = Pinecone(api_key="your-api-key")
index = pc.Index("my-rag-index")

# 创建索引
pc.create_index(
    name="my-rag-index",
    dimension=768,
    metric="cosine"
)

# 添加向量
vectors = [{"id": "doc1", "values": [0.1] * 768, "metadata": {"text": "内容"}}]
index.upsert(vectors)

# 搜索
results = index.query(
    vector=[0.1] * 768,
    top_k=5,
    include_metadata=True
)
```

## 4. Qdrant

### 特点
- 开源
- 高性能
- 支持过滤

### 使用

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient(host="localhost", port=6333)

# 创建集合
client.recreate_collection(
    collection_name="rag_docs",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
)

# 添加向量
client.upsert(
    collection_name="rag_docs",
    points=[
        {
            "id": 1,
            "vector": [0.1] * 768,
            "payload": {"text": "RAG是检索增强生成"}
        }
    ]
)

# 搜索
results = client.search(
    collection_name="rag_docs",
    query_vector=[0.1] * 768,
    limit=5
)
```

## 5. ChromaDB

### 特点
- 轻量级
- 纯Python
- 适合快速原型

### 使用

```python
import chromadb
from chromadb.utils import embedding_functions

# 初始化
client = chromadb.PersistentClient(path="./chroma_db")

# 使用sentence-transformers作为embedding函数
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-base-zh-v1.5"
)

# 创建集合
collection = client.get_or_create_collection(
    name="rag_docs",
    embedding_function=embedding_fn
)

# 添加文档
collection.add(
    documents=["RAG是检索增强生成", "Transformer是深度学习模型"],
    ids=["doc1", "doc2"]
)

# 搜索
results = collection.query(
    query_texts=["什么是RAG?"],
    n_results=5
)
```

## 6. 选择建议

| 场景 | 推荐 |
|------|------|
| 快速原型/学习 | ChromaDB |
| 研究/小规模实验 | Faiss |
| 中小规模生产 | Qdrant / Weaviate |
| 大规模云部署 | Pinecone / Milvus Cloud |

## 下一步

完成本模块后，进入 [1-foundations/README.md](../1-foundations/README.md) 学习RAG基础。
