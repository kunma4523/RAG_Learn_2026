# 生成模块

生成(Generation)是RAG的最后一步，利用增强后的上下文让LLM生成回答。

## 1. 生成流程

```
┌─────────────────────────────────────────────────────────────┐
│                      生成流程                                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  输入:                                                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Prompt with Context:                                │   │
│  │                                                     │   │
│  │ Context:                                            │   │
│  │ [Document 1] RAG是检索增强生成...                   │   │
│  │                                                     │   │
│  │ Question: 什么是RAG?                                │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                           ▼                                 │
│                    ┌─────────────┐                         │
│                    │     LLM     │                         │
│                    │  (GPT-4等)  │                         │
│                    └──────┬──────┘                         │
│                           │                                 │
│                           ▼                                 │
│  输出:                                                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ RAG（检索增强生成）是一种将信息检索与文本生成       │   │
│  │ 相结合的技术。它通过从外部知识库检索相关信息，        │   │
│  │ 并将检索结果作为上下文提供给大语言模型，从而         │   │
│  │ 提高生成内容的准确性和可信度。                       │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 2. LLM调用

### 2.1 OpenAI API

```python
from openai import OpenAI

client = OpenAI(api_key="your-key")

def generate(prompt, model="gpt-4"):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=500
    )
    return response.choices[0].message.content
```

### 2.2 HuggingFace 本地模型

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)

def generate(prompt):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=500,
        temperature=0.7
    )
    
    return tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], 
        skip_special_tokens=True
    )
```

## 3. 生成参数调优

| 参数 | 说明 | 建议值 |
|------|------|--------|
| temperature | 随机性，0-2 | 0.7 |
| top_p | 核采样 | 0.9 |
| max_tokens | 最大生成长度 | 500-1000 |
| presence_penalty | 避免重复 | 0-0.5 |

```python
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.7,        # 控制随机性
    top_p=0.9,             # 核采样
    max_tokens=500,        # 最大生成长度
    frequency_penalty=0.0, # 减少重复词
    presence_penalty=0.0   # 减少重复话题
)
```

## 4. 流式输出

```python
# 流式输出 - 逐字显示
def generate_stream(prompt):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    
    for chunk in response:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="")
```

## 5. 生成优化技巧

| 技巧 | 说明 |
|------|------|
|few-shot | 添加示例 |
| 思考链 | 添加"让我们分析" |
| 角色设定 | 指定AI角色 |
| 输出格式 | 指定JSON等格式 |

## 下一步

- [LangChain入门](./6-langchain-intro.md) - 学习LangChain
