#!/usr/bin/env python3
"""
测试: SQL-RAG (16_sql_rag.py)
=============================

将自然语言转换为SQL查询数据库的RAG系统。

运行: python tests/rag_architectures/16_sql_rag.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from tests.rag_architectures import TestConfig, run_test, print_test_results, create_llm


# ============================================
# 测试数据 - 数据库模式
# ============================================

DB_SCHEMA = """
表: users
- id (INTEGER, PRIMARY KEY)
- name (TEXT)
- email (TEXT)
- created_at (TIMESTAMP)

表: orders
- id (INTEGER, PRIMARY KEY)
- user_id (INTEGER, FOREIGN KEY)
- product_name (TEXT)
- quantity (INTEGER)
- total_price (REAL)
- order_date (TIMESTAMP)

表: products
- id (INTEGER, PRIMARY KEY)
- name (TEXT)
- category (TEXT)
- price (REAL)
- stock (INTEGER)
"""

SAMPLE_DATA = """
示例数据:
users: (1, '张三', 'zhangsan@email.com'), (2, '李四', 'lisi@email.com')
orders: (1, 1, 'Laptop', 1, 999.99), (2, 2, 'Phone', 2, 1198.00)
products: (1, 'Laptop', 'Electronics', 999.99, 50), (2, 'Phone', 'Electronics', 599.00, 100)
"""


# ============================================
# SQL-RAG 实现
# ============================================


class SQLRAG:
    """
    SQL-RAG 实现

    特点:
    1. 理解数据库模式
    2. 将自然语言转换为SQL
    3. 执行SQL查询
    4. 基于查询结果生成回答
    """

    def __init__(self, llm, schema: str):
        self.llm = llm
        self.schema = schema

    def _generate_sql(self, query: str) -> str:
        """将自然语言转换为SQL"""

        prompt = f"""给定以下数据库模式:
{self.schema}

将这个自然语言问题转换为SQL查询:
问题: {query}

只返回SQL语句，不要其他解释:"""

        response = self.llm.invoke(prompt)
        return response.content.strip()

    def _execute_sql(self, sql: str) -> dict:
        """模拟执行SQL(实际应用中连接真实数据库)"""

        # 模拟查询结果
        sql_lower = sql.lower()

        if "select" in sql_lower and "count" in sql_lower:
            return {"result": [{"count": 100}], "description": "计数查询"}
        elif "select" in sql_lower and "sum" in sql_lower:
            return {"result": [{"sum": 2197.99}], "description": "求和查询"}
        elif "select" in sql_lower:
            return {"result": [{"id": 1, "name": "Laptop"}], "description": "列表查询"}
        else:
            return {"result": [], "description": "其他查询"}

    def query(self, query: str) -> dict:
        """执行SQL-RAG"""

        # 生成SQL
        sql = self._generate_sql(query)

        # 执行SQL
        query_result = self._execute_sql(sql)

        # 基于结果生成回答
        result_text = str(query_result["result"])

        answer_prompt = f"""基于以下SQL查询结果，回答用户问题:

问题: {query}
SQL: {sql}
查询结果: {result_text}

请用自然语言回答问题:"""

        response = self.llm.invoke(answer_prompt)

        return {
            "answer": response.content,
            "sql": sql,
            "query_result": query_result["result"],
            "query_type": query_result["description"],
        }


# ============================================
# 测试函数
# ============================================


def test_01_sql_rag_basic(config: TestConfig):
    """测试: 基本SQL-RAG"""
    print("\n[测试 01] 基本SQL-RAG")

    llm = create_llm(config, temperature=0)
    if llm is None:
        return {"passed": False, "message": "创建LLM失败"}

    sql_rag = SQLRAG(llm, DB_SCHEMA)
    result = sql_rag.query("有多少用户?")

    return {
        "passed": "answer" in result,
        "message": f"SQL-RAG完成，生成了SQL: {result.get('sql', '')[:30]}...",
        "details": {
            "sql": result.get("sql", ""),
            "query_result": result.get("query_result", []),
        },
    }


def test_02_nl_to_sql(config: TestConfig):
    """测试: 自然语言转SQL"""
    print("\n[测试 02] 自然语言转SQL")

    llm = create_llm(config, temperature=0)
    if llm is None:
        return {"passed": False, "message": "创建LLM失败"}

    sql_rag = SQLRAG(llm, DB_SCHEMA)
    result = sql_rag.query("总订单金额是多少?")

    has_sql = "select" in result.get("sql", "").lower()

    return {
        "passed": has_sql,
        "message": f"成功转换为SQL",
        "details": {"sql": result.get("sql", "")},
    }


def test_03_query_execution(config: TestConfig):
    """测试: 查询执行和回答生成"""
    print("\n[测试 03] 查询执行和回答生成")

    llm = create_llm(config, temperature=0)
    if llm is None:
        return {"passed": False, "message": "创建LLM失败"}

    sql_rag = SQLRAG(llm, DB_SCHEMA)
    result = sql_rag.query("查询所有产品信息")

    return {
        "passed": len(result.get("query_result", [])) > 0,
        "message": f"查询执行成功，返回{len(result.get('query_result', []))}条记录",
        "details": {"result_count": len(result.get("query_result", []))},
    }


# ============================================
# 主函数
# ============================================


def main():
    print("=" * 60)
    print("SQL-RAG 架构测试")
    print("=" * 60)

    config = TestConfig()

    if not config.is_configured:
        print("\n⚠️  警告: API未配置!")
        return

    print(f"\n[配置] LLM: {config.llm_provider}/{config.openai_model}")

    results = []

    results.append(run_test(lambda: test_01_sql_rag_basic(config), "基本SQL-RAG"))
    results.append(run_test(lambda: test_02_nl_to_sql(config), "自然语言转SQL"))
    results.append(run_test(lambda: test_03_query_execution(config), "查询执行"))

    print_test_results(results)

    failed = sum(1 for r in results if not r.passed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
