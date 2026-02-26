# 提示压缩

提示压缩减少上下文长度，提高效率。

## 1. 方法

- 关键词提取
- 句子压缩
- 选择性保留

## 2. 实现

```python
# 使用LLMCompacter
from langchain_compression import LLMLingua

compressor = LLMLingua()
compressed_prompt = compressor.compress_prompt(long_prompt)
```

## 3. 注意事项

- 保留关键信息
- 平衡压缩率与效果
