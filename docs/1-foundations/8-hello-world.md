# Hello World示例

让我们创建一个最简单的RAG应用。

## 1. 环境准备

```bash
pip install langchain-core sentence-transformers faiss-cpu
```

## 2. 完整代码

```python
"""
最小RAG示例
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ===== 1. 准备知识库 =====
knowledge_base = [
    "RAG是Retrieval-Augmented Generation的缩写，中文名检索增强生成。",
    "RAG于2020年由Meta研究院提出。",
    "RAG可以解决大语言模型知识过时、幻觉等问题。",
    "RAG的核心组件包括检索器、增强器、生成器。",
    "常用的RAG框架有LangChain和LlamaIndex。",
    "向量数据库用于存储文档的向量表示。",
    "常用的embedding模型有BGE、OpenAI text-embedding-ada-002等。",
]

# ===== 2. 创建检索器 =====
class SimpleRetriever:
    def __init__(self, model_name="BAAI/bge-base-zh-v1.5"):
        print(f"加载模型: {model_name}")
        self.model = SentenceTransformer(model_name)
        
    def index(self, documents):
        print(f"索引 {len(documents)} 个文档")
        self.documents = documents
        self.embeddings = self.model.encode(documents)
        
    def retrieve(self, query, top_k=3):
        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # 排序
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [
            {"text": self.documents[i], "score": float(similarities[i])}
            for i in top_indices
        ]

# ===== 3. 创建生成器 (模拟) =====
class SimpleGenerator:
    def __init__(self):
        print("初始化生成器")
        
    def generate(self, query, context_docs):
        # 模拟生成
        context = "\n".join([d["text"] for d in context_docs])
        
        answer = f"""根据检索到的资料，回答关于"{query}"的问题：

参考内容：
{context}

回答：
这是关于{query}的说明。RAG（检索增强生成）是一种结合信息检索和文本生成的技术。它通过从知识库中检索相关信息，并将其作为上下文提供给大语言模型，从而提高生成内容的准确性和可靠性。

RAG的核心优势：
1. 可以访问最新知识
2. 减少幻觉
3. 提供答案来源
"""
        return answer

# ===== 4. 组装RAG系统 =====
class SimpleRAG:
    def __init__(self):
        self.retriever = SimpleRetriever()
        self.generator = SimpleGenerator()
        
    def setup(self, knowledge_base):
        self.retriever.index(knowledge_base)
        
    def query(self, question):
        # 检索
        docs = self.retriever.retrieve(question)
        
        # 生成
        answer = self.generator.generate(question, docs)
        
        return {
            "answer": answer,
            "sources": docs
        }

# ===== 5. 运行 =====
if __name__ == "__main__":
    # 创建RAG系统
    rag = SimpleRAG()
    
    # 准备知识库
    rag.setup(knowledge_base)
    
    # 问答
    question = "什么是RAG?"
    result = rag.query(question)
    
    print("=" * 50)
    print(f"问题: {question}")
    print("=" * 50)
    print(f"回答: {result['answer']}")
    print("=" * 50)
    print("参考文档:")
    for i, doc in enumerate(result['sources'], 1):
        print(f"  {i}. {doc['text'][:50]}... (分数: {doc['score']:.4f})")
```

## 3. 运行结果

```
加载模型: BAAI/bge-base-zh-v1.5
索引 7 个文档
初始化生成器
==================================================
问题: 什么是RAG?
==================================================
回答: 根据检索到的资料，回答关于"什么是RAG?"的问题：

参考内容：
RAG是Retrieval-Augmented Generation的缩写，中文名检索增强生成。
RAG可以解决大语言模型知识过时、幻觉等问题。
RAG的核心组件包括检索器、增强器、生成器。

回答：
这是关于什么是RAG?的说明。RAG（检索增强生成）是一种结合信息检索和文本生成的技术...
==================================================
参考文档:
  1. RAG是Retrieval-Augmented Generation的缩写，中文名检索增强生成。 (分数: 0.8542)
  2. RAG可以解决大语言模型知识过时、幻觉等问题。 (分数: 0.7823)
  3. RAG的核心组件包括检索器、增强器、生成器。 (分数: 0.7654)
```

## 4. 进阶：使用真实LLM

```python
# 替换SimpleGenerator为真实LLM
from openai import OpenAI

class OpenAIGenerator:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        
    def generate(self, query, context_docs):
        context = "\n".join([f"[{i+1}] {d['text']}" 
                           for i, d in enumerate(context_docs)])
        
        prompt = f"""基于以下参考资料回答问题。

参考资料：
{context}

问题：{query}

回答："""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content
```

## 下一步

进入 [2-architectures/README.md](../2-architectures/README.md) 学习各种RAG架构。
