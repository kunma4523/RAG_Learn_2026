# Self-Mem

**架构名称**: Self-Mem  
**类型**: 记忆增强型  
**难度**: ⭐⭐⭐  

## 1. 核心思想

结合外部检索与内部记忆：
1. 维护对话历史记忆
2. 检索历史中相关内容
3. 结合外部文档生成

## 2. 与Conversational RAG的区别

| 特性 | Conversational RAG | Self-Mem |
|------|------------------|----------|
| 记忆形式 | 对话历史 | 摘要/嵌入 |
| 检索方式 | 简单回顾 | 语义检索 |
| 适用 | 多轮对话 | 长对话 |

## 7. 代码示例

```python
# 运行测试脚本
python tests/rag_architectures/20_self_mem.py
```

或查看完整代码:
- 测试代码: `tests/rag_architectures/20_self_mem.py`
- 共享工具: `tests/rag_architectures/__init__.py`

## 8. 参考文献

- [Self-Mem Paper](https://arxiv.org/abs/2401.13075)
- [Self-Mem GitHub](https://github.com/StonyBrookNLP/simlm)