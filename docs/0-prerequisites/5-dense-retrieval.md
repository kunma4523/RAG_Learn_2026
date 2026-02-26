# 密集检索

密集检索(Dense Retrieval)使用神经网络将文本编码为密集向量，通过向量相似度进行检索。

## 1. 原理

```
┌─────────────────────────────────────────────────────────────┐
│                    密集检索流程                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐    ┌─────────────┐    ┌──────────────┐     │
│  │  Query   │───▶│  Encoder   │───▶│  Query Vec  │     │
│  │   "什么是RAG"│   │ (Transformer)│   │  [0.1, -0.3...] │     │
│  └──────────┘    └─────────────┘    └──────┬───────┘     │
│                                            │              │
│                                            ▼              │
│  ┌──────────┐    ┌─────────────┐    ┌──────────────┐     │
│  │ 文档集合  │───▶│  Encoder   │───▶│  Doc Vecs   │     │
│  │ Doc1     │   │ (Transformer)│   │  存储在索引中 │     │
│  │ Doc2     │   │             │   │              │     │
│  │ Doc3     │   │             │   │              │     │
│  └──────────┘    └─────────────┘    └──────┬───────┘     │
│                                            │              │
│                                            ▼              │
│                                   ┌──────────────┐        │
│                                   │  向量相似度  │        │
│                                   │  计算 & 排序  │        │
│                                   └──────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

## 2. 常用模型

### 2.1 Sentence Transformers

```python
from sentence_transformers import SentenceTransformer

# 加载模型
model = SentenceTransformer('BAAI/bge-base-zh-v1.5')

# 编码文档
doc_embeddings = model.encode(documents)

# 编码查询
query_embedding = model.encode([query])

# 计算相似度
from sklearn.metrics.pairwise import cosine_similarity
scores = cosine_similarity(query_embedding, doc_embeddings)
```

### 2.2 DPR (Dense Passage Retrieval)

```python
from transformers import DPRContextEncoder, DPRQuestionEncoder

# 加载编码器
ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
q_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

# 编码
import torch
with torch.no_grad():
    doc_emb = ctx_encoder(**tokenizer(documents, return_tensors="pt", padding=True))["pooler_output"]
    query_emb = q_encoder(**tokenizer(query, return_tensors="pt"))["pooler_output"]
```

## 3. 训练方法

### 3.1 对比学习

```python
from sentence_transformers import SentenceTransformerTrainer
from sentence_transformers import ContrastiveLoss

# 使用对比损失训练
loss = ContrastiveLoss(model)
trainer = SentenceTransformerTrainer(
    model=model,
    train_dataset=train_dataset,
    loss=loss,
)
trainer.train()
```

### 3.2 难负样本挖掘

```python
# In-batch negatives + 硬负样本
# 使用批次内负样本和难负样本训练
```

## 4. 常用模型对比

| 模型 | 语言 | 维度 | 特点 |
|------|------|------|------|
| bge-base-zh-v1.5 | 中英 | 768 | 效果好，开源 |
| bge-large-zh-v1.5 | 中英 | 1024 | 效果最好 |
| text-embedding-ada-002 | 多语言 | 1536 | OpenAI API |
| E5 | 多语言 | 768 | 英文效果好 |

## 5. 实战

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 初始化
model = SentenceTransformer('BAAI/bge-base-zh-v1.5')

# 文档库
documents = [
    "RAG是检索增强生成技术",
    "Transformer是一种深度学习架构",
    "BERT是预训练语言模型",
]

# 编码
doc_embeddings = model.encode(documents)

# 检索函数
def retrieve(query, top_k=2):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [(documents[i], similarities[i]) for i in top_indices]

# 测试
results = retrieve("什么是RAG?")
for doc, score in results:
    print(f"分数: {score:.4f} | 文档: {doc}")
```

## 下一步

- [向量数据库](./6-vector-db.md) - 学习向量存储与检索
