# In-Context RALM

**架构名称**: In-Context RALM  
**类型**: 上下文学习型  
**难度**: ⭐⭐  

## 1. 核心思想

In-Context RALM不需要微调：
- 在prompt中直接包含检索结果
- 模型通过few-shot学习利用上下文
- 适合任何LLM

HP|| 实现难度 | 简单 | 复杂 |

## 3. 代码示例

```python
# 运行测试脚本
python tests/rag_architectures/12_in_context_ralm.py
```

或查看完整代码:
- 测试代码: `tests/rag_architectures/12_in_context_ralm.py`

| 特性 | In-Context RALM | Self-RAG |
|------|-----------------|----------|
| 微调 | 不需要 | 需要 |
| 反思token | 不使用 | 使用 |
| 实现难度 | 简单 | 复杂 |
