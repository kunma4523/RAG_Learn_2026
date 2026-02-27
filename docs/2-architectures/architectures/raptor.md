# RAPTOR

**架构名称**: RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval)  
**类型**: 摘要树型  
**难度**: ⭐⭐⭐  

## 1. 核心思想

构建文档摘要树：
1. 底层：原始文档块
2. 递归聚类，生成摘要
3. 逐层检索直到找到答案

## 2. 架构

```
                    [Root Summary]
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
     [Summary A]    [Summary B]    [Summary C]
          │              │              │
    ┌─────┴─────┐  ┌─────┴─────┐  ┌─────┴─────┐
    ▼           ▼  ▼           ▼  ▼           ▼
  [Doc1]     [Doc2] [Doc3]   [Doc4] [Doc5]   [Doc6]
```

## 3. 论文

- [RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval](https://arxiv.org/abs/2401.xxxxx)

## 4. 代码示例

```python
# 运行测试脚本
python tests/rag_architectures/21_raptor.py
```

或查看完整代码:
- 测试代码: `tests/rag_architectures/21_raptor.py`

- [RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval](https://arxiv.org/abs/2401.xxxxx)
