# Transformer原理

Transformer是一种基于自注意力机制的深度学习架构，2017年由Vaswani等人提出，已成为NLP领域的主流模型基础。

## 1. 核心架构

```
┌─────────────────────────────────────────────────────────────┐
│                     Transformer 架构                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐                                              │
│  │  Input   │                                              │
│  │  Embedding│                                             │
│  └────┬─────┘                                              │
│       │                                                    │
│       ▼                                                    │
│  ┌─────────────────────┐                                   │
│  │   Positional Encoding │                                 │
│  └──────────┬───────────┘                                   │
│             │                                               │
│             ▼                                               │
│  ┌─────────────────────┐     ┌─────────────────────┐      │
│  │   Encoder Stack    │     │   Decoder Stack    │      │
│  │                    │     │                    │      │
│  │  • Multi-Head Att │     │  • Multi-Head Att  │      │
│  │  • Feed Forward   │     │  • Feed Forward    │      │
│  │  • Add & Norm    │     │  • Add & Norm      │      │
│  └─────────────────────┘     └──────────┬──────────┘      │
│                                        │                  │
│                                        ▼                  │
│                               ┌─────────────────┐         │
│                               │   Linear + Softmax│        │
│                               └─────────────────┘         │
│                                        │                  │
│                                        ▼                  │
│                               ┌─────────────────┐         │
│                               │    Output      │         │
│                               └─────────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

## 2. 注意力机制

### 自注意力公式

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

其中：
- $Q$ (Query): 查询向量
- $K$ (Key): 键向量
- $V$ (Value): 值向量
- $d_k$: 键向量的维度

### 多头注意力

```
┌────────────────────────────────────────┐
│           Multi-Head Attention          │
├────────────────────────────────────────┤
│                                        │
│  Q ──▶ Linear ──▶ Head 1 ──┐         │
│       Linear ──▶ Head 2 ──┼──▶ Concat ──▶ Linear ──▶ Output
│       ...                  │         │
│  Q ──▶ Linear ──▶ Head h ──┘         │
│                                        │
└────────────────────────────────────────┘
```

## 3. 关键组件

### 位置编码 (Positional Encoding)

```python
import numpy as np

def positional_encoding(position, d_model):
    # 偶数位置：sin
    # 奇数位置：cos
    angle_rads = np.arange(position)[:, np.newaxis] / np.power(10000, (2 * (np.arange(d_model) // 2)) / d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return angle_rads
```

### 前馈网络 (Feed Forward)

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))
```

## 4. 使用Transformers库

```python
from transformers import AutoTokenizer, AutoModel

# 加载预训练模型
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = AutoModel.from_pretrained("bert-base-chinese")

# 编码输入
inputs = tokenizer("Hello, world!", return_tensors="pt")
outputs = model(**inputs)
```

## 5. Transformer变体

| 模型 | 特点 | 适用场景 |
|------|------|----------|
| BERT | 双向编码 | 文本理解 |
| GPT | 单向解码 | 文本生成 |
| T5 | Encoder-Decoder | 序列到序列 |
| ViT | Vision Transformer | 图像处理 |

## 下一步

- [Prompt工程](./2-prompt-engineering.md) - 学习提示词设计
