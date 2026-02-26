# LlamaIndex入门

LlamaIndex（原GPT-Index）是专注于数据增强的LLM框架。

## 1. 核心概念

```
┌─────────────────────────────────────────────────────────────┐
│                  LlamaIndex 核心组件                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐                                          │
│  │ Data Loaders │  │ 100+ 数据源支持                        │
│  └─────────────┘                                          │
│        │                                                   │
│        ▼                                                   │
│  ┌─────────────┐                                          │
│  │   Indices   │  │ 向量索引、树索引、关键词索引            │
│  └─────────────┘                                          │
│        │                                                   │
│        ▼                                                   │
│  ┌─────────────┐                                          │
│  │  Query     │  │ Retriever, QueryEngine                │
│  └─────────────┘                                          │
│        │                                                   │
│        ▼                                                   │
│  ┌─────────────┐                                          │
│  │  Response  │  │ 生成器                                │
│  └─────────────┘                                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 2. 快速开始

```python
from llama_index.core import SimpleDirectoryReader

# 1. 加载文档
documents = SimpleDirectoryReader("./data").load_data()

# 2. 创建索引
from llama_index.core import VectorStoreIndex

index = VectorStoreIndex.from_documents(documents)

# 3. 创建查询引擎
query_engine = index.as_query_engine()

# 4. 查询
response = query_engine.query("什么是RAG?")
print(response)
```

## 3. 核心组件

### 3.1 文档加载

```python
from llama_index.core import SimpleDirectoryReader

# 加载目录
loader = SimpleDirectoryReader("./data", recursive=True)
docs = loader.load_data()

# 指定文件
loader = SimpleDirectoryReader(
    input_files=["file1.pdf", "file2.txt"]
)
docs = loader.load_data()

# 多种格式支持
# PDF, TXT, MD, DOCX, Notion, Discord, etc.
```

### 3.2 索引类型

```python
from llama_index.core import (
    VectorStoreIndex,      # 向量索引
    TreeIndex,           # 树索引
    KeywordTableIndex,    # 关键词索引
    SummaryIndex,        # 摘要索引
)

# 向量索引 (最常用)
vector_index = VectorStoreIndex.from_documents(docs)

# 树索引 (适合总结)
tree_index = TreeIndex.from_documents(docs)

# 关键词索引
keyword_index = KeywordTableIndex.from_documents(docs)
```

### 3.3 查询引擎

```python
# 基础查询
query_engine = index.as_query_engine()

# 带相似度阈值
query_engine = index.as_query_engine(
    similarity_top_k=5,
    similarity_cutoff=0.7
)

# 带重排序
from llama_index.core.postprocessor import SentenceEmbeddingReranker

reranker = SentenceEmbeddingReranker(
    rerank_top_n=3,
    model="BAAI/bge-reranker-base"
)

query_engine = index.as_query_engine(
    node_postprocessors=[reranker]
)
```

## 4. 高级功能

### 4.1 自定义分块

```python
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter

parser = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=50,
    include_metadata=True
)

nodes = parser.get_nodes_from_documents(docs)
index = VectorStoreIndex(nodes)
```

### 4.2 混合检索

```python
from llama_index.core.retrievers import (
    VectorRetriever,
    BM25Retriever
)
from llama_index.core import VectorStoreIndex

# 向量检索
vector_retriever = VectorStoreIndex.as_retriever(
    similarity_top_k=10
)

# BM25检索
bm25_retriever = BM25Retriever.from_defaults(
    docstore=index.docstore,
    similarity_top_k=10
)

# 混合
from llama_index.core.retrievers import QueryFusionRetriever

retriever = QueryFusionRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    mode="reciprocal_rank_fusion"
)
```

### 4.3 Response Synthesis

```python
from llama_index.core import get_response_synthesizer

# 创建响应合成器
synthesizer = get_response_synthesizer(
    response_mode="compact",  # compact, tree_summarize, simple
    text_qa_template=prompt
)

# 使用
query_engine = index.as_query_engine(
    response_synthesizer=synthesizer
)
```

## 5. LangChain vs LlamaIndex

| 特性 | LangChain | LlamaIndex |
|------|-----------|------------|
| 定位 | 通用LLM框架 | 数据增强优先 |
| 索引 | 基础 | 丰富(树、关键词等) |
| 查询 | Chain | QueryEngine |
| 文档处理 | 简单 | 强大 |
| 学习曲线 | 中等 | 较低 |

## 6. 完整示例

```python
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 配置embedding
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-base-zh-v1.5"
)

# 1. 加载
docs = SimpleDirectoryReader("./data").load_data()

# 2. 索引
index = VectorStoreIndex.from_documents(docs)

# 3. 查询
query_engine = index.as_query_engine(
    similarity_top_k=3
)

response = query_engine.query("你的问题")
print(response)
```

## 下一步

- [Hello World示例](./8-hello-world.md) - 第一个RAG应用
