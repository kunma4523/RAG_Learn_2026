# SQL-RAG

**架构名称**: SQL-RAG  
**类型**: 结构化数据型  
**难度**: ⭐⭐⭐  

## 1. 核心思想

将自然语言转为SQL查询数据库：
1. 用户问题 → SQL
2. 执行SQL获取结果
3. 基于结果生成回答

## 2. 工作流程

```
┌─────────────────────────────────────────────┐
│              SQL-RAG                          │
├─────────────────────────────────────────────┤
│                                             │
│  Question: "2024年销售额最高的产品是什么?"   │
│              │                               │
│              ▼                               │
│  ┌─────────────────────────────────────┐   │
│  │  SQL Generation (LLM)               │   │
│  │  SELECT product_name FROM sales     │   │
│  │  WHERE year = 2024                  │   │
│  │  ORDER BY sales DESC LIMIT 1        │   │
│  └─────────────────────────────────────┘   │
│              │                               │
│              ▼                               │
│  ┌─────────────────────────────────────┐   │
│  │  Execute SQL                        │   │
│  │  结果: [{"product": "A产品", ...}]  │   │
│  └─────────────────────────────────────┘   │
│              │                               │
│              ▼                               │
│  Answer: "2024年销售额最高的产品是A产品"    │
└─────────────────────────────────────────────┘
```

## 3. 实现

(代码示例见下方)

## 4. 代码示例

```python
# 运行测试脚本
python tests/rag_architectures/16_sql_rag.py
```

或查看完整代码:
- 测试代码: `tests/rag_architectures/16_sql_rag.py`

```python
from langchain.agents import create_sql_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit

toolkit = SQLDatabaseToolkit(db=dbllm)
agent, llm= = create_sql_agent(llm, toolkit=toolkit)
```
