# LangChain入门

LangChain是一个简化LLM应用开发的Python框架。

## 1. 核心概念

```
┌─────────────────────────────────────────────────────────────┐
│                   LangChain 核心组件                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐                                          │
│  │   Models   │  │  LLM, Chat Models, Embeddings        │
│  └─────────────┘                                          │
│        │                                                   │
│        ▼                                                   │
│  ┌─────────────┐                                          │
│  │ Prompts   │  │  PromptTemplates, OutputParsers        │
│  └─────────────┘                                          │
│        │                                                   │
│        ▼                                                   │
│  ┌─────────────┐                                          │
│  │   Chains   │  │  LCEL, Sequential, Router            │
│  └─────────────┘                                          │
│        │                                                   │
│        ▼                                                   │
│  ┌─────────────┐                                          │
│  │   Agents   │  │  Tool use, Action selection          │
│  └─────────────┘                                          │
│        │                                                   │
│        ▼                                                   │
│  ┌─────────────┐                                          │
│  │   Memory   │  │  Chat history, Context                │
│  └─────────────┘                                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 2. 基础RAG实现

```python
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain import hub
from langchain_core.runnables import RunnablePassthrough

# 1. 加载文档
loader = TextLoader("document.txt")
documents = loader.load()

# 2. 文档分块
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)

# 3. 创建向量存储
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-zh-v1.5")
vectorstore = FAISS.from_documents(chunks, embeddings)

# 4. 创建检索器
retriever = vectorstore.as_retriever()

# 5. 定义Prompt
prompt = hub.pull("rlm/rag-prompt")

# 6. 构建Chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 7. 使用
response = chain.invoke("什么是RAG?")
print(response)
```

## 3. LCEL语法

```python
# LangChain Expression Language (LCEL)
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# 基础
chain = prompt | llm | output_parser

# 并行
chain = RunnableParallel(
    context=retriever,
    question=RunnablePassthrough()
) | prompt | llm

# 条件分支
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template("...")
llm = ChatOpenAI(model="gpt-4")

# 带缓存
from langchain.cache import InMemoryCache
import langchain
langchain.llm_cache = InMemoryCache()

chain = prompt | llm
```

## 4. 常用组件

### 4.1 文档加载

```python
# 多种加载器
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    CSVLoader,
    WebLoader
)

# PDF
loader = PyPDFLoader("file.pdf")
docs = loader.load()

# 网页
loader = WebLoader("https://example.com")
docs = loader.load()
```

### 4.2 文本分割

```python
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter
)

# 递归分割 (推荐)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", " "]
)
```

### 4.3 向量存储

```python
from langchain_community.vectorstores import (
    FAISS,
    Chroma,
    Pinecone,
    Milvus
)

# FAISS (本地)
vectorstore = FAISS.from_documents(docs, embeddings)

# Chroma
vectorstore = Chroma.from_documents(docs, embeddings)

# Pinecone (云)
vectorstore = Pinecone.from_documents(docs, embeddings, index_name="my-index")
```

## 5. RAG Chain完整示例

```python
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser

# 初始化
llm = ChatOpenAI(model="gpt-4", temperature=0)

# 使用LangChain Hub的RAG prompt
prompt = hub.pull("rlm/rag-prompt")

# 构建chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 查询
result = rag_chain.invoke("你的问题")
```

## 下一步

- [LlamaIndex入门](./7-llamaindex-intro.md) - 学习LlamaIndex
