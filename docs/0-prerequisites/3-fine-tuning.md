# 模型微调

模型微调是指在预训练模型基础上，使用特定数据集进一步训练，使模型适应特定任务。

## 1. 微调类型

```
┌─────────────────────────────────────────────────────────────┐
│                    模型训练 vs 微调                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  预训练 (Pre-training)                                     │
│  ┌─────────────┐                                           │
│  │ 海量文本数据 │ ──▶ 基础语言模型                         │
│  └─────────────┘                                           │
│                                                             │
│  微调 (Fine-tuning)                                        │
│  ┌─────────────┐                                           │
│  │ 预训练模型   │ + 特定数据 ──▶ 专用模型                  │
│  └─────────────┘                                           │
│                                                             │
│  两种微调方式：                                             │
│  1. 全参数微调 (Full Fine-tuning)                         │
│  2. 参数高效微调 (PEFT)                                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 2. 全参数微调

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 准备数据
from datasets import load_dataset
dataset = load_dataset("your_dataset")

# 训练
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=5e-5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
)

trainer.train()
```

## 3. 参数高效微调 (PEFT)

### 3.1 LoRA

```python
from peft import LoraConfig, get_peft_model

# 配置LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
)

# 应用LoRA
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 8,388,608 || all params: 124,879,872 || trainable%: 6.72
```

### 3.2 QLoRA

```python
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

# 4-bit量化配置
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",
)

# 加载量化模型
model = AutoModelForCausalLM.from_pretrained(
    "model_name",
    quantization_config=quantization_config,
)

# 准备训练
model = prepare_model_for_kbit_training(model)
```

### 3.3 Prefix Tuning

```python
from peft import PrefixTuningConfig

prefix_config = PrefixTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=20,
)
```

## 4. RAG中的微调

### 4.1 检索器微调

```python
# 使用对比学习微调检索器
from sentence_transformers import SentenceTransformerTrainer
from sentence_transformers import ContrastiveLoss

# 加载预训练模型
retriever = SentenceTransformer('BAAI/bge-base-zh-v1.5')

# 准备训练数据
train_dataset = ...

# 定义损失函数
loss = ContrastiveLoss(model)

# 训练
trainer = SentenceTransformerTrainer(
    model=retriever,
    train_dataset=train_dataset,
    loss=loss,
)
trainer.train()
```

### 4.2 生成器微调

```python
# 微调LLM用于RAG场景
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=512,
)
trainer.train()
```

## 5. 微调最佳实践

| 阶段 | 建议 |
|------|------|
| 数据准备 | 确保数据质量，500-1000条即可开始 |
| 学习率 | 一般1e-4 到 1e-5 |
| Epochs | 2-5个epoch足够 |
| 评估 | 使用验证集监控过拟合 |
| 保存 | 保存最佳checkpoint |

## 6. 何时微调vs使用Prompt

```
┌─────────────────────────────────────────┐
│           决策流程                        │
├─────────────────────────────────────────┤
│                                         │
│  任务是否需要特定格式？ ──是──▶ 提示工程   │
│       │                                  │
│       否                                 │
│       ▼                                 │
│  任务是否需要学习新知识？ ──否──▶ 提示工程   │
│       │                                  │
│       是                                 │
│       ▼                                 │
│  有足够的训练数据？ ──否──▶ RAG           │
│       │                                  │
│       是                                 │
│       ▼                                 │
│  考虑微调 + RAG                          │
│                                         │
└─────────────────────────────────────────┘
```

## 下一步

- [传统检索方法](./4-traditional-retrieval.md) - 学习TF-IDF、BM25
