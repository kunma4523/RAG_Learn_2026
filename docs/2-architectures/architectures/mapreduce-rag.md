# MapReduce RAG

**架构名称**: MapReduce RAG  
**类型**: 长文档处理型  
**难度**: ⭐⭐  

## 1. 核心思想

分而治之处理长文档：
1. **Map**: 将长文档分块，每块生成摘要
2. **Reduce**: 合并摘要，最终生成

## 2. 工作流程

```
长文档 (10000字)
    │
    ▼
┌───────────────────────────────────┐
│  Split: 拆分成多个chunk           │
│  [chunk1, chunk2, ..., chunkN]    │
└───────────────────────────────────┘
    │
    ├──▶ chunk1 ──▶ 摘要1
    ├──▶ chunk2 ──▶ 摘要2
    ├──▶ ... 
    └──▶ chunkN ──▶ 摘要N
    
    │
    ▼
┌───────────────────────────────────┐
│  Reduce: 合并摘要                │
│  摘要1 + 摘要2 + ...            │
└───────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────┐
│  Final LLM: 基于合并摘要生成答案  │
└───────────────────────────────────┘
```

## 3. 实现

```python
from langchain.chains import MapReduceDocumentsChain

map_chain = LLMChain(llm=llm, prompt=map_prompt)
reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

map_reduce = MapReduceDocumentsChain(
    llm_chain=map_chain,
    reduce_chain=reduce_chain,
    document_variable_name="context"
)
```
