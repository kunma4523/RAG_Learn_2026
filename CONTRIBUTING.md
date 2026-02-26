# 贡献指南

感谢您对 RAG-Learning-2026 项目的兴趣！我们欢迎任何形式的贡献，包括但不限于：

- 🐛 修复错误
- 📚 添加新的RAG架构
- 💻 改进现有代码
- 📝 撰写或改进教程文档
- 🔧 添加新的实验配置
- 📊 改进评估指标

## 如何贡献

### 1. Fork 项目

点击页面右上Fork" 按钮角的 "。

### 2. 克隆本地

```bash
git clone https://github.com/YOUR_USERNAME/RAG-Learning-2026.git
cd RAG-Learning-2026
```

### 3. 创建分支

```bash
git checkout -b feature/your-feature-name
# 或
git checkout -b fix/bug-description
```

### 4. 进行修改

请确保：
- 遵循项目的代码风格
- 添加适当的注释
- 更新相关文档

### 5. 提交更改

```bash
git add .
git commit -m "feat: 添加 xxx 功能"
```

提交信息格式：
- `feat`: 新功能
- `fix`: 修复bug
- `docs`: 文档更新
- `refactor`: 代码重构
- `test`: 测试相关
- `chore`: 构建/工具链变动

### 6. 推送分支

```bash
git push origin feature/your-feature-name
```

### 7. 创建 Pull Request

在 GitHub 上创建 PR，请包含：
- 清晰的标题和描述
- 关联的 Issue（如果有）
- 必要的测试结果

## 项目结构说明

```
RAG-Learning-2026/
├── docs/           # 文档（Markdown格式）
├── notebooks/      # Jupyter Notebook
├── src/            # 可复用代码
│   ├── retrievers/ # 检索器
│   ├── generators/ # 生成器
│   ├── pipelines/  # RAG流程
│   └── evaluation/ # 评估
├── experiments/    # 实验配置
└── data/          # 示例数据
```

## 代码规范

### Python
- 遵循 [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- 使用 type hints
- 使用 docstrings 描述函数/类

### 文档
- 使用中文或英文（保持一致性）
- 代码示例需要可运行
- 复杂概念需要图解或示例说明

## 行为准则

请阅读并遵守我们的 [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)（如有）。

## 问题与讨论

- 🐛 发现Bug？请创建 Issue
- 💡 有新想法？请创建 Feature Request
- ❓ 有问题？可以使用 Discussions

## 感谢

感谢所有贡献者！🎉
