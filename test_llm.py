#!/usr/bin/env python
"""
LLM 生成器测试
==============

测试 LLM 生成器是否正常工作。
直接运行: python test_llm.py

注意: 运行前请确保已安装依赖并在 .env 文件中配置好 API Key
"""

import sys
import os

# 将项目根目录添加到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.generators.llm import (
    get_llm_generator,
    OpenAIGenerator,
    AnthropicGenerator,
    GoogleGenerator,
    QwenGenerator,
    HuggingFaceGenerator,
)
from src.utils.config import get_llm_provider, get_llm_config


def test_llm_generator():
    """
    测试根据环境配置自动选择 LLM 生成器

    Returns:
        bool: 测试是否成功
    """
    print("=" * 60)
    print("测试 LLM 生成器自动选择")
    print("=" * 60)

    # 打印当前配置
    provider = get_llm_provider()
    config = get_llm_config()

    print(f"\n当前 LLM 提供商: {provider}")
    print(f"模型: {config.get('model', 'N/A')}")
    print()

    # 创建生成器
    try:
        generator = get_llm_generator()
        print(f"✓ 成功创建生成器: {generator}")
    except Exception as e:
        print(f"✗ 创建生成器失败: {e}")
        return False

    # 测试生成
    test_prompt = "请用一句话介绍你自己"

    print(f"\n测试提示: {test_prompt}")
    print("-" * 40)

    try:
        result = generator.generate(test_prompt)

        if result.text:
            print(f"\n✓ 生成成功!")
            print(f"  响应: {result.text[:200]}...")
            print(f"  元数据: {result.metadata}")
            return True
        else:
            print(f"\n✗ 生成失败")
            print(f"  错误: {result.metadata.get('error', 'Unknown')}")
            return False

    except Exception as e:
        print(f"\n✗ 生成异常: {e}")
        return False


def test_specific_generator():
    """
    测试特定的生成器（如果配置了相应的 API Key）

    Returns:
        bool: 测试是否成功
    """
    print("\n" + "=" * 60)
    print("测试特定生成器")
    print("=" * 60)

    config = get_llm_config()
    api_key = config.get("api_key")

    if not api_key or api_key == "your-api-key-here" or api_key == "sk-your-key-here":
        print("\n⚠ 未配置 API Key，跳过特定生成器测试")
        print("  请在 .env 文件中配置相应的 API Key")
        return True

    providers_tested = 0

    # 测试 OpenAI
    if config.get("provider") == "openai":
        print("\n测试 OpenAI 生成器...")
        try:
            gen = OpenAIGenerator()
            result = gen.generate("你好，请介绍一下你自己")
            print(f"  ✓ OpenAI 响应: {result.text[:100]}...")
            providers_tested += 1
        except Exception as e:
            print(f"  ✗ OpenAI 测试失败: {e}")

    # 测试 Anthropic
    if config.get("provider") == "anthropic":
        print("\n测试 Anthropic 生成器...")
        try:
            gen = AnthropicGenerator()
            result = gen.generate("你好，请介绍一下你自己")
            print(f"  ✓ Anthropic 响应: {result.text[:100]}...")
            providers_tested += 1
        except Exception as e:
            print(f"  ✗ Anthropic 测试失败: {e}")

    # 测试 Google
    if config.get("provider") == "google":
        print("\n测试 Google 生成器...")
        try:
            gen = GoogleGenerator()
            result = gen.generate("你好，请介绍一下你自己")
            print(f"  ✓ Google 响应: {result.text[:100]}...")
            providers_tested += 1
        except Exception as e:
            print(f"  ✗ Google 测试失败: {e}")

    # 测试 Qwen
    if config.get("provider") == "qwen":
        print("\n测试 Qwen 生成器...")
        try:
            gen = QwenGenerator()
            result = gen.generate("你好，请介绍一下你自己")
            print(f"  ✓ Qwen 响应: {result.text[:100]}...")
            providers_tested += 1
        except Exception as e:
            print(f"  ✗ Qwen 测试失败: {e}")

    return providers_tested > 0


def test_batch_generate():
    """
    测试批量生成功能

    Returns:
        bool: 测试是否成功
    """
    print("\n" + "=" * 60)
    print("测试批量生成")
    print("=" * 60)

    try:
        generator = get_llm_generator()

        prompts = [
            "什么是人工智能?",
            "什么是机器学习?",
            "什么是深度学习?",
        ]

        print(f"\n测试 {len(prompts)} 个提示...")

        results = generator.batch_generate(prompts)

        success_count = sum(1 for r in results if r.text)

        print(f"\n✓ 批量生成完成: {success_count}/{len(prompts)} 成功")

        for i, result in enumerate(results):
            print(f"\n  [{i + 1}] {prompts[i]}")
            print(f"      响应: {result.text[:80]}...")

        return success_count == len(prompts)

    except Exception as e:
        print(f"\n✗ 批量生成测试失败: {e}")
        return False


def test_create_prompt():
    """
    测试提示创建功能

    Returns:
        bool: 测试是否成功
    """
    print("\n" + "=" * 60)
    print("测试提示创建")
    print("=" * 60)

    try:
        generator = get_llm_generator()

        query = "什么是 RAG?"
        context = [
            "RAG 是 Retrieval-Augmented Generation 的缩写，中文名为检索增强生成。",
            "RAG 可以帮助大模型访问最新信息和专业知识。",
        ]

        prompt = generator.create_prompt(query, context)

        print(f"\n查询: {query}")
        print(f"\n生成的提示:\n{prompt}")

        # 测试带系统提示的版本
        prompt_with_system = generator.create_prompt(
            query, context, system_prompt="你是一个专业的 AI 助手。"
        )

        print(f"\n带系统提示的版本:\n{prompt_with_system}")

        return True

    except Exception as e:
        print(f"\n✗ 提示创建测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("       RAG-Learning LLM 生成器测试")
    print("=" * 60)
    print("\n本测试将验证 LLM 生成器配置是否正确")
    print("请确保已在 .env 文件中配置好 API Key\n")

    # 检查环境配置
    print("检查环境配置...")
    provider = get_llm_provider()
    config = get_llm_config()

    print(f"  LLM 提供商: {provider}")
    print(f"  模型: {config.get('model', 'N/A')}")

    # 检查 API Key
    api_key = config.get("api_key")
    if not api_key or api_key in ["your-api-key-here", "sk-your-key-here"]:
        print("\n⚠ 警告: 未在 .env 中配置有效的 API Key")
        print("  测试可能会失败，请配置后再试")

    print()

    # 运行测试
    all_passed = True

    # 测试 1: 自动选择生成器
    all_passed &= test_llm_generator()

    # 测试 2: 批量生成
    all_passed &= test_batch_generate()

    # 测试 3: 提示创建
    all_passed &= test_create_prompt()

    # 测试 4: 特定生成器
    all_passed &= test_specific_generator()

    # 总结
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ 所有测试通过!")
    else:
        print("✗ 部分测试失败，请检查配置")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
