# 传统检索方法

传统信息检索方法主要包括TF-IDF和BM25，是RAG系统中稀疏检索的基础。

## 1. TF-IDF

### 原理

TF-IDF (Term Frequency-Inverse Document Frequency) 是一种统计方法，用于评估一个词对一个文档集的重要程度。

### 计算公式

$$TF(t,d) = \frac{\text{词t在文档d中出现的次数}}{\text{文档d的总词数}}$$

$$IDF(t) = \log\left(\frac{\text{文档总数}}{\text{包含词t的文档数}}\right)$$

$$TF-IDF(t,d) = TF(t,d) \times IDF(t)$$

### 实现

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# 创建TF-IDF向量化器
vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words='english',
    ngram_range=(1, 2)  # 使用1-gram和2-gram
)

# 拟合并转换文档
tfidf_matrix = vectorizer.fit_transform(documents)

# 转换查询
query_vector = vectorizer.transform([query])

# 计算相似度
from sklearn.metrics.pairwise import cosine_similarity
scores = cosine_similarity(query_vector, tfidf_matrix)
```

### 优缺点

| 优点 | 缺点 |
|------|------|
| 简单快速 | 无法处理同义词 |
| 可解释性强 | 无法捕捉语义 |
| 无需训练 | 对词形变化不敏感 |

## 2. BM25

### 原理

BM25 (Best Matching 25) 是TF-IDF的改进版本，是Elasticsearch等搜索引擎的默认算法。

### 计算公式

$$Score(Q,d) = \sum_{i=1}^{n} IDF(q_i) \cdot \frac{f(q_i,D) \cdot (k_1+1)}{f(q_i,D) + k_1 \cdot (1 - b + b \cdot \frac{|D|}{avgdl})}$$

其中：
- $f(q_i,D)$: 词 $q_i$ 在文档 $D$ 中的频率
- $|D|$: 文档长度
- $avgdl$: 平均文档长度
- $k_1$: 词频饱和参数 (通常1.2-2.0)
- $b$: 文档长度归一化参数 (通常0.75)

### 实现

```python
from rank_bm25 import BM25Okapi
import re

# 分词
def tokenize(text):
    return re.findall(r'\w+', text.lower())

# 准备语料库
tokenized_corpus = [tokenize(doc) for doc in documents]

# 创建BM25模型
bm25 = BM25Okapi(tokenized_corpus)

# 查询
tokenized_query = tokenize(query)
scores = bm25.get_scores(tokenized_query)

# 获取top-k结果
top_docs = bm25.get_top_n(tokenized_query, documents, n=5)
```

### 优缺点

| 优点 | 缺点 |
|------|------|
| 词频饱和 | 仍无法处理同义词 |
| 文档长度归一化 | 稀疏表示 |
| 效果好，稳定 | 难以处理语义相似 |

## 3. 对比

| 特性 | TF-IDF | BM25 |
|------|---------|------|
| 词频处理 | 线性 | 非线性(饱和) |
| 文档长度 | 未归一化 | 已归一化 |
| 实际效果 | 一般 | 更好 |
| 计算复杂度 | O(n) | O(n) |

## 4. 混合使用

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import numpy as np

class HybridRetrieval:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.tfidf = TfidfVectorizer()
        self.bm25 = None
    
    def fit(self, documents):
        # TF-IDF
        self.tfidf.fit(documents)
        self.tfidf_matrix = self.tfidf.transform(documents)
        
        # BM25
        tokenized = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)
        
        self.documents = documents
    
    def search(self, query, top_k=5):
        # TF-IDF分数
        tfidf_query = self.tfidf.transform([query])
        tfidf_scores = cosine_similarity(tfidf_query, self.tfidf_matrix).flatten()
        
        # BM25分数
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # 归一化
        tfidf_scores = tfidf_scores / (tfidf_scores.max() + 1e-8)
        bm25_scores = bm25_scores / (bm25_scores.max() + 1e-8)
        
        # 融合
        combined = self.alpha * tfidf_scores + (1-self.alpha) * bm25_scores
        
        # 返回top-k
        top_indices = combined.argsort()[-top_k:][::-1]
        return [(self.documents[i], combined[i]) for i in top_indices]
```

## 下一步

- [密集检索](./5-dense-retrieval.md) - 学习基于向量的检索
