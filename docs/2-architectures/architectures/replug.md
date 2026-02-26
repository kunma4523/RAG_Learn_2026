# REPLUG

**架构名称**: REPLUG  
**类型**: 迭代检索型  
**难度**: ⭐⭐⭐  

## 1. 核心思想

REPLUG将检索器与LLM解耦，通过多次检索迭代改进结果：
1. LLM生成初始回答
2. 根据回答中的线索检索更多文档
3. 迭代改进直到收敛

## 2. 特点

- 检索器独立于LLM
- 使用加权组合多个检索结果
- 适合复杂推理任务

## 3. 论文

- [REPLUG: Retrieval-Augmented Black-Box Language Models](https://arxiv.org/abs/2301.xxxxx)
