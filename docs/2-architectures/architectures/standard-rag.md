# Standard RAG

**架构名称**: Standard RAG  
**类型**: 基础型  
**难度**: ⭐  

## 1. 论文/来源

- **论文**: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
- **作者**: Patrick Lewis et al.
- **年份**: 2020
- **arXiv**: [2005.11401](https://arxiv.org/abs/2005.11401)

## 2. 核心思想

标准RAG是最基础的RAG架构，将检索、增强、生成三个步骤串联：
1. **检索**: 使用用户查询从知识库中检索相关文档
2. **增强**: 将检索到的文档作为上下文拼接到提示词中
3. **生成**: 大语言模型基于增强后的提示词生成回答

## 3. 架构图

```
┌──────────┐    ┌─────────────┐    ┌──────────────┐    ┌──────────────┐
│   Query  │───▶│  Retrieval  │───▶│ Augmentation │───▶│  Generation  │
│   用户查询 │    │    检索     │    │     增强     │    │     生成     │
└──────────┘    └─────────────┘    └──────────────┘    └──────────────┘
                                                              │
                                                              ▼
                                                     ┌──────────────┐
                                                     │    Answer    │
                                                     │    答案      │
                                                     └──────────────┘
```

## 4. 适用场景

- 简单的问答系统
- 文档问答
- 知识库查询

## 5. 优点

- 实现简单，易于理解
- 效果好，广泛使用
- 可扩展性强

## 6. 缺点

- 只能单次检索，不适合复杂推理
- 无法处理多轮对话
- 检索质量直接影响生成效果

## 7. 代码示例

```python
from src.pipelines import StandardRAGPipeline

pipeline = StandardRAGPipeline(
    embedding_model="BAAI/bge-base-zh-v1.5",
    llm_model="gpt-4"
)

# 索引文档
pipeline.index_documents(documents)

# 问答
result = pipeline.query("什么是RAG?")
print(result.answer)
```

## 8. 参考文献

- [RAG Paper](https://arxiv.org/abs/2005.11401)
- [LangChain RAG](https://python.langchain.com/docs/use_cases/question_answering/)
