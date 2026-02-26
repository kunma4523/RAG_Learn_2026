# 模型量化

压缩模型，降低资源需求。

## 1. 量化方法

- FP16: 半精度
- INT8: 8位整数
- GPTQ: 动态量化
- AWQ: 激活感知量化

## 2. 实现

```python
# bitsandbytes
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16"
)
```
