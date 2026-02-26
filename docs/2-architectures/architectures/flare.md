# FLARE

**架构名称**: FLARE (Forward-Looking Active Retrieval Augmented Generation)  
**类型**: 动态检索型  
**难度**: ⭐⭐⭐  

## 1. 核心思想

FLARE在生成过程中主动触发检索：
1. 生成一小段
2. 如果有不确信的内容，检索补充
3. 继续生成

## 2. 工作流程

```
生成: "RAG是..."
    │
    ├── 检测: 是否需要检索?
    ├── 是 → 检索相关文档
    │       重新生成
    └── 否 → 继续生成
    
生成完整后返回
```

## 3. 论文

- [FLARE: Active Retrieval Augmented Generation](https://arxiv.org/abs/2303.xxxxx)
