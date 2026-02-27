#!/usr/bin/env python3
"""
测试: Table RAG (17_table_rag.py)
==================================

针对表格数据的RAG系统，能够理解和推理表格结构。

运行: python tests/rag_architectures/17_table_rag.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from tests.rag_architectures import (
    TestConfig,
    run_test,
    print_test_results,
    create_vector_store,
    create_llm,
)


# ============================================
# 测试数据 - 表格
# ============================================

TABLES = {
    "products": {
        "headers": ["产品名称", "类别", "价格", "库存", "销量"],
        "rows": [
            ["Laptop", "电子产品", "999.99", "50", "120"],
            ["Phone", "电子产品", "599.00", "100", "200"],
            ["Tablet", "电子产品", "399.00", "75", "80"],
            ["Desk", "家具", "299.00", "30", "25"],
            ["Chair", "家具", "199.00", "40", "35"],
        ],
    },
    "sales": {
        "headers": ["月份", "产品", "销售额", "销售量"],
        "rows": [
            ["1月", "Laptop", "29999.70", "30"],
            ["1月", "Phone", "35940.00", "60"],
            ["2月", "Laptop", "19999.80", "20"],
            ["2月", "Phone", "23960.00", "40"],
        ],
    },
}


# ============================================
# Table RAG 实现
# ============================================


class TableRAG:
    """
    Table RAG 实现

    特点:
    1. 理解表格结构
    2. 表格内容检索
    3. 结构化数据分析
    4. 基于表格的回答
    """

    def __init__(self, llm, tables: dict):
        self.llm = llm
        self.tables = tables

    def _format_table(self, table_name: str) -> str:
        """格式化表格为文本"""

        table = self.tables.get(table_name, {})
        headers = table.get("headers", [])
        rows = table.get("rows", [])

        lines = [f"表: {table_name}"]
        lines.append(" | ".join(headers))
        lines.append("-" * 30)

        for row in rows:
            lines.append(" | ".join(str(cell) for cell in row))

        return "\n".join(lines)

    def _extract_relevant_tables(self, query: str) -> list:
        """提取相关表格"""

        table_summaries = "\n".join(
            [f"- {name}: {', '.join(t['headers'])}" for name, t in self.tables.items()]
        )

        prompt = f"""用户问题: {query}

可用表格:
{table_summaries}

哪些表格与回答这个问题相关? 返回表格名称，用逗号分隔:"""

        response = self.llm.invoke(prompt)

        relevant = []
        for name in self.tables.keys():
            if name in response.content:
                relevant.append(name)

        return relevant if relevant else list(self.tables.keys())[:1]

    def query(self, query: str) -> dict:
        """执行Table RAG"""

        # 找到相关表格
        relevant_tables = self._extract_relevant_tables(query)

        # 格式化表格内容
        table_contents = []
        for table_name in relevant_tables:
            table_contents.append(self._format_table(table_name))

        combined_context = "\n\n".join(table_contents)

        prompt = f"""基于以下表格数据回答问题:

{combined_context}

问题: {query}

请基于表格数据给出准确回答:"""

        response = self.llm.invoke(prompt)

        return {
            "answer": response.content,
            "relevant_tables": relevant_tables,
            "table_contents": table_contents,
        }


# ============================================
# 测试函数
# ============================================


def test_01_table_rag_basic(config: TestConfig):
    """测试: 基本Table RAG"""
    print("\n[测试 01] 基本Table RAG")

    llm = create_llm(config, temperature=0)
    if llm is None:
        return {"passed": False, "message": "创建LLM失败"}

    table_rag = TableRAG(llm, TABLES)
    result = table_rag.query("最贵的产品是什么?")

    return {
        "passed": "answer" in result,
        "message": f"Table RAG完成，分析了{len(result.get('relevant_tables', []))}个表格",
        "details": {
            "tables_used": result.get("relevant_tables", []),
            "answer_preview": result.get("answer", "")[:100],
        },
    }


def test_02_table_structure_understanding(config: TestConfig):
    """测试: 表格结构理解"""
    print("\n[测试 02] 表格结构理解")

    llm = create_llm(config, temperature=0)
    if llm is None:
        return {"passed": False, "message": "创建LLM失败"}

    table_rag = TableRAG(llm, TABLES)
    result = table_rag.query("电子产品的总库存是多少?")

    return {
        "passed": "answer" in result,
        "message": f"识别了相关表格: {result.get('relevant_tables', [])}",
        "details": {"tables": result.get("relevant_tables", [])},
    }


def test_03_aggregation_queries(config: TestConfig):
    """测试: 聚合查询"""
    print("\n[测试 03] 聚合查询")

    llm = create_llm(config, temperature=0)
    if llm is None:
        return {"passed": False, "message": "创建LLM失败"}

    table_rag = TableRAG(llm, TABLES)
    result = table_rag.query("计算Laptop的总销售额")

    return {
        "passed": "answer" in result,
        "message": "聚合查询完成",
        "details": {"answer_preview": result.get("answer", "")[:100]},
    }


# ============================================
# 主函数
# ============================================


def main():
    print("=" * 60)
    print("Table RAG 架构测试")
    print("=" * 60)

    config = TestConfig()

    if not config.is_configured:
        print("\n⚠️  警告: API未配置!")
        return

    print(f"\n[配置] LLM: {config.llm_provider}/{config.openai_model}")

    results = []

    results.append(run_test(lambda: test_01_table_rag_basic(config), "基本Table RAG"))
    results.append(
        run_test(lambda: test_02_table_structure_understanding(config), "表格结构理解")
    )
    results.append(run_test(lambda: test_03_aggregation_queries(config), "聚合查询"))

    print_test_results(results)

    failed = sum(1 for r in results if not r.passed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
