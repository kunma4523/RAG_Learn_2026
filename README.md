# RAG-Learning-2026

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python"></a>
  <a href="#"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
  <a href="#"><img src="https://img.shields.io/badge/Docs-81-brightgreen.svg" alt="Docs"></a>
</p>

<p align="center">
  <h3>🚀 面向2026年的RAG系统性学习开源项目</h3>
</p>

本项目整合了经典与前沿的RAG（检索增强生成）架构，提供理论讲解、代码实现、实验评估与前沿追踪，帮助学习者从零构建对RAG的全面认知，并具备实际应用与创新能力。

## ✨ 特性

- 📚 **系统性教程** - 从基础到高级，循序渐进的RAG知识体系
- 🏗️ **23+ RAG架构** - 涵盖经典9种+补充14种主流架构
- 💻 **完整代码实现** - 基于LangChain/LlamaIndex的实战代码
- 📓 **Jupyter Notebook** - 交互式Notebook实战
- 🧪 **实验框架** - 可复现的实验配置与评估工具
- 🔄 **持续更新** - 追踪2025-2026最新顶会论文

## 📁 项目结构

```
RAG-Learning-2026/
├── docs/                    # 文档教程 (81个文件)
│   ├── 0-prerequisites/     # 预备知识
│   ├── 1-foundations/       # RAG基础
│   ├── 2-architectures/     # 各架构详解
│   ├── 3-advanced/          # 高级专题
│   └── 4-future/            # 前沿与趋势
├── notebooks/               # Jupyter Notebook 实战
├── src/                     # 可复用的代码库
│   ├── retrievers/          # 检索器实现
│   ├── generators/          # 生成器封装
│   ├── pipelines/           # 各架构流程
│   └── evaluation/          # 评估指标与工具
├── experiments/             # 实验脚本与配置
├── data/                    # 示例数据（或数据下载脚本）
├── references/              # 论文、博客等参考资料
├── contributions/           # 社区贡献区
├── README.md                # 项目主页
├── LICENSE                  # 开源协议
└── CONTRIBUTING.md          # 贡献指南
```

## 📚 学习模块

| 模块 | 内容 | 状态 |
|------|------|------|
| [0-prerequisites](./docs/0-prerequisites/README.md) | 预备知识 (Python、ML基础、检索技术) | ✅ 已完成 |
| [1-foundations](./docs/1-foundations/README.md) | RAG基础概念与流程 | ✅ 已完成 |
| [2-architectures](./docs/2-architectures/README.md) | 23种RAG架构详解 | ✅ 已完成 |
| [3-advanced](./docs/3-advanced/README.md) | 高级专题 (优化、评估、部署) | ✅ 已完成 |
| [4-future](./docs/4-future/README.md) | 前沿趋势 (2025-2026) | ✅ 已完成 |

## 🗺️ 学习路径

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAG-Learning-2026                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐       │
│   │  预备知识    │───▶│  RAG基础    │───▶│  经典架构   │       │
│   │ Prerequisites│    │ Foundations │    │ Architectures│      │
│   └─────────────┘    └─────────────┘    └─────────────┘       │
│                                                │                │
│                                                ▼                │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐       │
│   │  前沿趋势    │◀───│  高级专题   │◀───│  补充架构   │       │
│   │   Future    │    │  Advanced   │    │ More Archs  │       │
│   └─────────────┘    └─────────────┘    └─────────────┘       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/your-username/RAG-Learning-2026.git
cd RAG-Learning-2026
```

### 2. 创建虚拟环境

```bash
# 使用 uv (推荐)
uv venv
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate     # Windows

# 或使用 conda
conda create -n rag-learn python=3.10
conda activate rag-learn
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 运行示例

```python
from src.pipelines import StandardRAGPipeline

# 初始化pipeline
pipeline = StandardRAGPipeline(
    embedding_model="BAAI/bge-base-zh-v1.5",
    llm_model="Qwen/Qwen2-7B-Instruct"
)

# 索引文档
pipeline.index_documents(["RAG文档内容..."])

# 问答
answer = pipeline.query("什么是RAG?")
print(answer.answer)
```

## 📊 文档统计

| 模块 | 文件数 | 状态 |
|------|--------|------|
| 0-预备知识 | 8 | ✅ |
| 1-RAG基础 | 8 | ✅ |
| 2-RAG架构 | 23 | ✅ |
| 3-高级专题 | 16 | ✅ |
| 4-前沿趋势 | 20 | ✅ |
| **总计** | **81** | ✅ **全部完成** |

## 🤝 贡献指南

欢迎任何形式的贡献！请阅读 [CONTRIBUTING.md](CONTRIBUTING.md) 了解如何参与贡献。

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- [LangChain](https://github.com/langchain-ai/langchain)
- [LlamaIndex](https://github.com/run-llama/llama_index)
- [RAGAS](https://github.com/explodinggradients/ragas)
- 所有贡献者和社区成员

---

<p align="center">
  <sub>Built with ❤️ for the RAG community</sub>
</p>
