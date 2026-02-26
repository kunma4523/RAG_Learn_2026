# 增强模块

增强(Augmentation)将检索到的文档与查询结合，形成LLM可理解的上下文。

## 1. 增强流程

```
┌─────────────────────────────────────────────────────────────┐
│                      增强流程                                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  输入:                                                      │
│  ┌──────────┐  ┌─────────────┐                            │
│  │  Query   │  │ Retrieved   │                            │
│  │ "什么是RAG?"│  │ Docs [1,2,3] │                            │
│  └──────────┘  └─────────────┘                            │
│                                                             │
│  处理:                                                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 1. 文档切分 (按长度/段落)                           │   │
│  │ 2. 上下文窗口选择 (最近的K个)                        │   │
│  │ 3. 格式化为Prompt                                   │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  输出:                                                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Context:                                            │   │
│  │ [Document 1] RAG是检索增强生成...                   │   │
│  │                                                     │   │
│  │ [Document 2] RAG于2020年由Meta提出...              │   │
│  │                                                     │   │
│  │ Question: 什么是RAG?                                │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 2. 上下文窗口管理

### 2.1 固定窗口

```python
def create_context_fixed(docs, max_tokens=2000):
    """固定窗口大小"""
    context = ""
    for doc in docs:
        if len(context) + len(doc) > max_tokens:
            break
        context += f"[Document {len(context.split('Document'))}]\n{doc}\n\n"
    return context
```

### 2.2 滑动窗口

```python
def create_context_sliding(docs, max_tokens=2000, overlap=100):
    """滑动窗口，保留重叠"""
    context = ""
    for i, doc in enumerate(docs):
        doc_tokens = doc.split()
        # 简单的token估算
        if len(context.split()) + len(doc_tokens) > max_tokens:
            # 添加重叠部分
            last_words = context.split()[-overlap:]
            context = " ".join(last_words) + " "
        
        context += f"[Document {i+1}] {doc}\n\n"
    return context
```

### 2.3 动态窗口

```python
def create_context_dynamic(docs, query, max_tokens=2000):
    """根据相关性动态选择"""
    # 按相关性排序
    sorted_docs = sorted(docs, key=lambda x: x["score"], reverse=True)
    
    context = ""
    for doc in sorted_docs:
        # 检查是否包含查询关键词
        if any(word in doc["text"] for word in query.split()):
            # 优先选择包含关键词的文档
            if len(context.split()) + len(doc["text"].split()) <= max_tokens:
                context += f"[Document]\n{doc['text']}\n\n"
    
    return context
```

## 3. Prompt模板

### 3.1 基础模板

```python
BASE_TEMPLATE = """基于以下参考资料回答问题。

参考资料：
{context}

问题：{question}

回答："""
```

### 3.2 详细模板

```python
DETAILED_TEMPLATE = """你是一个专业的问答助手。请根据以下参考资料回答用户问题。

要求：
1. 只根据参考资料回答，不要添加额外信息
2. 如果参考资料中没有相关信息，请说明"根据提供的信息无法回答"
3. 回答要简洁准确

参考资料：
{context}

用户问题：{question}

你的回答："""
```

### 3.3 带思考链的模板

```python
COT_TEMPLATE = """请根据以下参考资料回答问题。在回答前，先分析参考资料中的相关信息。

参考资料：
{context}

问题：{question}

分析：{question}涉及到...

根据参考资料：
1. ...
2. ...

因此答案是："""
```

## 4. 最佳实践

| 实践 | 说明 |
|------|------|
| 文档标识 | 给每个文档编号，方便引用 |
| 长度控制 | 不超过模型上下文限制 |
| 关键词高亮 | 可以用**加粗**关键信息 |
| 分隔清晰 | 用```隔离代码块 |

## 下一步

- [生成模块](./5-generation.md) - 学习LLM生成
