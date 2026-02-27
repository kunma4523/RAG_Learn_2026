# Conversational RAG

**架构名称**: Conversational RAG  
**类型**: 交互型  
**难度**: ⭐⭐  

## 1. 论文/来源

- 基于对话上下文的RAG系统
- 开源实现: LangChain ConversationRetrievalChain

## 2. 核心思想

对话式RAG在标准RAG基础上增加了对话历史管理：
1. 维护多轮对话历史
2. 将历史对话融入查询改写
3. 动态检索相关上下文

## 3. 架构图

```
┌──────────────┐     ┌─────────────┐
│ Conversation │────▶│ Query       │
│   History    │     │ Rewriting   │
└──────────────┘     └──────┬──────┘
                           │
                           ▼
┌──────────┐    ┌─────────────┐    ┌──────────────┐
│   Query  │───▶│  Retrieval  │───▶│ Augmentation │
│ + History│    │             │    │ + History    │
└──────────┘    └─────────────┘    └──────┬───────┘
                                         │
                                         ▼
                                ┌──────────────┐
                                │  Generation  │
                                └──────┬───────┘
                                       │
                                       ▼
                                ┌──────────────┐
                                │    Answer    │
                                └──────────────┘
```

## 4. 适用场景

- 聊天机器人
- 多轮问答系统
- 客服系统

## 5. 优点

- 支持多轮对话
- 理解对话上下文
- 用户体验好

## 6. 缺点

- 对话历史可能引入噪声
- 长对话占用更多token
- 历史处理策略影响效果

## 7. 代码示例

```python
# 运行测试脚本
python tests/rag_architectures/02_conversational_rag.py
```

或查看完整代码:
- 测试代码: `tests/rag_architectures/02_conversational_rag.py`

## 8. 参考文献

- [LangChain Conversation](https://python.langchain.com/docs/modules/chains/)
