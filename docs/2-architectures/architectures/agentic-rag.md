# Agentic RAG

**架构名称**: Agentic RAG  
**类型**: 智能体型  
**难度**: ⭐⭐⭐⭐  

## 1. 论文/来源

- 基于LLM Agent的RAG系统
- 参考实现: LangChain Agent, LlamaIndex Agent

## 2. 核心思想

Agentic RAG将LLM作为Agent，结合规划、推理能力：
1. **查询分析**: Agent理解用户意图
2. **动态规划**: 决定检索策略和工具使用
3. **迭代执行**: 多步检索和推理
4. **结果验证**: 验证答案的正确性

## 3. 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                     Agentic RAG                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐                                               │
│  │   User   │                                               │
│  │  Query   │                                               │
│  └────┬─────┘                                               │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────────────────┐                                    │
│  │      Planner        │                                    │
│  │   (LLM as Agent)    │                                    │
│  └──────────┬───────────┘                                    │
│             │                                               │
│    ┌────────┼────────┐                                       │
│    ▼        ▼        ▼                                      │
│ ┌────┐  ┌────┐  ┌─────┐                                    │
│ │Tool│  │Tool│  │Tool │  (Search, Calc, etc.)              │
│ │ 1  │  │ 2  │  │ 3   │                                    │
│ └──┬─┘  └──┬─┘  └──┬──┘                                    │
│    └───────┼───────┘                                       │
│            ▼                                                 │
│  ┌─────────────────────┐                                    │
│  │    Executor         │                                    │
│  │  (Orchestrates)     │                                    │
│  └──────────┬───────────┘                                    │
│             │                                               │
│             ▼                                               │
│  ┌─────────────────────┐                                    │
│  │      Verifier       │                                    │
│  │   (Self-Check)      │                                    │
│  └──────────┬───────────┘                                    │
│             │                                               │
│             ▼                                               │
│      ┌─────────────┐                                        │
│      │    Answer   │                                        │
│      └─────────────┘                                        │
└─────────────────────────────────────────────────────────────┘
```

## 4. 适用场景

- 复杂推理任务
- 多步问答
- 需要外部工具的任务
- 研究型问答

## 5. 优点

- 高度灵活
- 可处理复杂任务
- 可扩展性强

## 6. 缺点

- 实现复杂
- 调试困难
- 延迟较高

## 7. 代码示例

```python
# 运行测试脚本
python tests/rag_architectures/08_agentic_rag.py
```

或查看完整代码:
- 测试代码: `tests/rag_architectures/08_agentic_rag.py`

## 8. 参考文献

```python
from src.pipelines import AgenticRAGPipeline

pipeline = AgenticRAGPipeline(
    retriever=retriever,
    generator=generator,
    max_iterations=3,
    enable_verification=True
)

result = pipeline.query("查找2024年ACL最佳论文并总结其贡献")
```

## 8. 参考文献

- [LangChain Agents](https://python.langchain.com/docs/modules/agents/)
- [LlamaIndex Agents](https://docs.llamaindex.ai/en/stable/module_guides/querying/agent/)
