# Table RAG

**架构名称**: Table RAG  
**类型**: 表格型  
**难度**: ⭐⭐⭐  

## 1. 核心思想

针对表格数据的RAG：
1. 理解表格结构
2. 识别表格中的实体和关系
3. 基于表格内容回答问题

## 2. 适用场景

- 财务报表分析
- 统计数据分析
- Excel/CSV问答

## 3. 实现

(代码示例见下方)

## 4. 代码示例

```python
# 运行测试脚本
python tests/rag_architectures/17_table_rag.py
```

或查看完整代码:
- 测试代码: `tests/rag_architectures/17_table_rag.py`

```python
from langchain_experimental.tabular_synthesizer import TableRAG

table_rag = TableRAG(csv_file="data.csv")
answer = table_rag.query("哪一列的总和最高?")
```
