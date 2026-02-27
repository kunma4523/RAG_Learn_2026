# 配置管理模块
# 用于从 .env 文件加载配置

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 加载 .env 文件
load_dotenv(PROJECT_ROOT / ".env")


def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    获取环境变量值

    Args:
        key: 环境变量名称
        default: 默认值

    Returns:
        环境变量值或默认值
    """
    return os.getenv(key, default)


def get_llm_provider() -> str:
    """获取 LLM 提供商"""
    return get_env("LLM_PROVIDER", "openai").lower()


def get_llm_config() -> dict:
    """
    获取 LLM 配置

    Returns:
        包含 LLM 配置的字典
    """
    provider = get_llm_provider()

    config = {
        "provider": provider,
        "temperature": float(get_env("LLM_TEMPERATURE", "0.7")),
        "max_tokens": int(get_env("LLM_MAX_TOKENS", "2048")),
        "top_p": float(get_env("LLM_TOP_P", "0.9")),
    }

    if provider == "openai":
        config.update(
            {
                "api_key": get_env("OPENAI_API_KEY"),
                "base_url": get_env("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                "model": get_env("OPENAI_MODEL", "gpt-4o-mini"),
            }
        )
    elif provider == "anthropic":
        config.update(
            {
                "api_key": get_env("ANTHROPIC_API_KEY"),
                "model": get_env("ANTHROPIC_MODEL", "claude-3-haiku-20240307"),
            }
        )
    elif provider == "google":
        config.update(
            {
                "api_key": get_env("GOOGLE_API_KEY"),
                "model": get_env("GOOGLE_MODEL", "gemini-1.5-flash"),
            }
        )
    elif provider == "qwen":
        config.update(
            {
                "api_key": get_env("DASHSCOPE_API_KEY"),
                "model": get_env("QWEN_MODEL", "qwen-turbo"),
            }
        )
    elif provider == "local":
        config.update(
            {
                "model": get_env("LOCAL_LLM_MODEL", "Qwen/Qwen2-7B-Instruct"),
                "device": get_env("LOCAL_LLM_DEVICE", "cpu"),
            }
        )

    return config


def get_embedding_provider() -> str:
    """获取 Embedding 提供商"""
    return get_env("EMBEDDING_PROVIDER", "openai").lower()


def get_embedding_config() -> dict:
    """
    获取 Embedding 配置

    Returns:
        包含 Embedding 配置的字典
    """
    provider = get_embedding_provider()

    config = {
        "provider": provider,
        "batch_size": int(get_env("EMBEDDING_BATCH_SIZE", "32")),
        "top_k": int(get_env("EMBEDDING_TOP_K", "5")),
    }

    if provider == "openai":
        config.update(
            {
                "model": get_env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            }
        )
    elif provider == "dashscope":
        config.update(
            {
                "api_key": get_env("DASHSCOPE_API_KEY"),
                "model": get_env("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v3"),
            }
        )
    elif provider == "local":
        config.update(
            {
                "model": get_env("LOCAL_EMBEDDING_MODEL", "BAAI/bge-base-zh-v1.5"),
                "device": get_env("LOCAL_EMBEDDING_DEVICE", "cpu"),
            }
        )

    return config
