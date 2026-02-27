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
- 🏗️ **22+ RAG架构** - 涵盖经典9种+补充14种主流架构
- 💻 **完整代码实现** - 可直接运行的测试实例
- 🧪 **实验框架** - 每个架构都有可执行的测试代码
- 🔄 **持续更新** - 追踪2025-2026最新顶会论文

## 📁 项目结构

```
RAG-Learning-2026/
├── docs/                    # 文档教程
│   ├── 0-prerequisites/     # 预备知识
│   ├── 1-foundations/       # RAG基础
│   ├── 2-architectures/    # 22种架构详解
│   │   └── architectures/  # 各架构文档
│   ├── 3-advanced/          # 高级专题
│   └── 4-future/            # 前沿与趋势
├── tests/                   # 测试代码 (可直接运行)
│   └── rag_architectures/  # 22个架构测试脚本
│       ├── 01_standard_rag.py       # 标准RAG
│       ├── 02_conversational_rag.py # 对话式RAG
│       ├── 03_corrective_rag.py     # 纠正RAG
│       ├── 04_adaptive_rag.py       # 自适应RAG
│       ├── 05_self_rag.py           # 自反思RAG
│       ├── 06_fusion_rag.py         # 融合RAG
│       ├── 07_hyde.py               # HyDE
│       ├── 08_agentic_rag.py        # 智能体RAG
│       ├── 09_graph_rag.py          # 图RAG
│       ├── 10_replug.py             # REPLUG
│       ├── 11_iterative_rag.py      # 迭代RAG
│       ├── 12_in_context_ralm.py    # In-Context RALM
│       ├── 13_flare.py              # FLARE
│       ├── 14_multimodal_rag.py     # 多模态RAG
│       ├── 15_self_ask_rag.py       # Self-Ask + RAG
│       ├── 16_sql_rag.py            # SQL-RAG
│       ├── 17_table_rag.py         # Table RAG
│       ├── 18_mapreduce_rag.py      # MapReduce RAG
│       ├── 19_dragin.py             # DRAGIN
│       ├── 20_self_mem.py           # Self-Mem
│       ├── 21_raptor.py             # RAPTOR
│       ├── 22_ra_cm3.py             # RA-CM3
│       ├── __init__.py              # 测试工具
│       └── run_all_tests.py        # 运行所有测试
├── notebooks/               # Jupyter Notebook 实战
├── data/                    # 示例数据
├── references/              # 论文参考资料
├── README.md                # 项目主页
├── LICENSE                  # 开源协议
└── CONTRIBUTING.md          # 贡献指南
```

## 📚 学习模块

| 模块 | 内容 | 状态 |
|------|------|------|
| [0-prerequisites](./docs/0-prerequisites/README.md) | 预备知识 (Python、ML基础、检索技术) | ✅ |
| [1-foundations](./docs/1-foundations/README.md) | RAG基础概念与流程 | ✅ |
| [2-architectures](./docs/2-architectures/README.md) | 22种RAG架构详解 | ✅ |
| [3-advanced](./docs/3-advanced/README.md) | 高级专题 (优化、评估、部署) | ✅ |
| [4-future](./docs/4-future/README.md) | 前沿趋势 (2025-2026) | ✅ |

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone git@github.com:kunma4523/RAG_Learn_2026.git
cd RAG_Learn_2026
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

### 4. 配置API

复制 `.env.example` 为 `.env` 并配置API密钥:

```bash
cp .env.example .env
# 编辑 .env 填入你的 API key
```

### 5. 运行测试

```bash
# 运行单个架构测试
python tests/rag_architectures/01_standard_rag.py

# 运行所有22个架构测试
python tests/rag_architectures/run_all_tests.py

# 查看所有可用测试
python tests/rag_architectures/run_all_tests.py --list
```

## 📊 22种RAG架构测试

| 编号 | 测试文件 | 架构 | 描述 |
|------|----------|------|------|
| 01 | `01_standard_rag.py` | 标准RAG | 单次检索，拼接上下文生成 |
| 02 | `02_conversational_rag.py` | 对话式RAG | 维护对话历史，动态检索 |
| 03 | `03_corrective_rag.py` | 纠正RAG | 检索后验证、重写或补充 |
| 04 | `04_adaptive_rag.py` | 自适应RAG | 根据查询复杂度选择策略 |
| 05 | `05_self_rag.py` | 自反思RAG | 生成过程中自我反思与检索 |
| 06 | `06_fusion_rag.py` | 融合RAG | 多路检索结果融合排序 |
| 07 | `07_hyde.py` | HyDE | 先生成假设文档再检索 |
| 08 | `08_agentic_rag.py` | 智能体RAG | 结合Agent规划与工具调用 |
| 09 | `09_graph_rag.py` | 图RAG | 利用知识图谱增强检索 |
| 10 | `10_replug.py` | REPLUG | 检索器与LLM解耦，加权输入 |
| 11 | `11_iterative_rag.py` | 迭代RAG | 多跳检索与生成交替 |
| 12 | `12_in_context_ralm.py` | In-Context RALM | 不微调，仅靠上下文学习 |
| 13 | `13_flare.py` | FLARE | 生成中主动触发检索 |
| 14 | `14_multimodal_rag.py` | 多模态RAG | 图文音视频联合检索 |
| 15 | `15_self_ask_rag.py` | Self-Ask + RAG | 自问自答，逐层检索 |
| 16 | `16_sql_rag.py` | SQL-RAG | NL转SQL查询数据库 |
| 17 | `17_table_rag.py` | Table RAG | 针对表格的检索推理 |
| 18 | `18_mapreduce_rag.py` | MapReduce RAG | 分而治之，合并摘要 |
| 19 | `19_dragin.py` | DRAGIN | 根据注意力动态检索 |
| 20 | `20_self_mem.py` | Self-Mem | 结合外部检索与内部记忆 |
| 21 | `21_raptor.py` | RAPTOR | 构建文档摘要树，分层检索 |
| 22 | `22_ra_cm3.py` | RA-CM3 | 检索图文对指导图像生成 |

## 📊 文档统计

| 模块 | 文件数 | 状态 |
|------|--------|------|
| 0-预备知识 | 8 | ✅ |
| 1-RAG基础 | 8 | ✅ |
| 2-RAG架构 | 22+ | ✅ |
| 3-高级专题 | 16 | ✅ |
| 4-前沿趋势 | 20 | ✅ |
| **总计** | **81+** | ✅ |

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
