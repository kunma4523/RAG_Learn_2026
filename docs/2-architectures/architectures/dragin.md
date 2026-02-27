# DRAGIN

**架构名称**: DRAGIN  
**类型**: 注意力驱动型  
**难度**: ⭐⭐⭐⭐  

## 1. 核心思想

基于注意力动态决定检索时机：
1. 使用LLM的attention权重
2. 识别需要外部知识的token
3. 针对性地检索

## 2. 特点

- 细粒度检索控制
- 减少不必要的检索
- 高效利用计算资源

## 3. 论文

- [DRAGIN: Dynamic Retrieval Augmented Generation](https://arxiv.org/abs/2401.xxxxx)

## 4. 代码示例

```python
# 运行测试脚本
python tests/rag_architectures/19_dragin.py
```

或查看完整代码:
- 测试代码: `tests/rag_architectures/19_dragin.py`

- [DRAGIN: Dynamic Retrieval Augmented Generation](https://arxiv.org/abs/2401.xxxxx)
