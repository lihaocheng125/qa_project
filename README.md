# qa_project
实现一个本地的问答系统，可以根据人类反馈调整答案。

## 功能介绍

### 1. 问答系统核心功能
- **词法分析**：使用 `jieba` 库对用户输入的问题进行词法分析，将问题进行分词处理，以便后续处理和理解用户的问题意图。
- **问答模型**：结合规则匹配和预训练的问答模型（`bert-large-uncased-whole-word-masking-finetuned-squad`）来生成问题的答案。如果问题在已有的问答对中存在，则直接返回对应的答案；否则，利用预训练模型进行推理，根据模型的输出得到答案。若模型未能找到合适的答案，会提示用户提供更多反馈。

### 2. 人类反馈机制
- **反馈收集**：用户可以对系统给出的回答进行反馈，反馈信息包括问题、答案以及用户的满意度（满意/不满意）。这些反馈数据会被存储在 `feedback_data` 列表中，方便后续分析和处理。
- **答案更新**：当用户反馈为“不满意”时，系统会提示用户提供正确的答案，然后将该问题和用户提供的新答案更新到问答对 `qa_pairs` 中。下次遇到相同问题时，系统将使用用户提供的新答案进行回答。

### 3. 前端交互
- **问题输入与提交**：用户可以在前端页面的输入框中输入问题，点击“提问”按钮后，通过 AJAX 技术将问题发送到后端服务器，等待服务器返回分词结果和系统回答。
- **反馈提交**：在得到系统回答后，用户可以对回答进行满意度评价，并点击“提交反馈”按钮，将反馈信息通过 AJAX 发送到后端服务器，服务器会根据反馈信息进行相应处理，并将处理结果返回给前端显示。

### 4. 项目配置与管理
- **Django 框架**：整个项目基于 Django 框架构建，利用 Django 的 URL 路由、视图函数、模板系统等功能，实现了前后端的交互和数据处理。
- **环境配置**：通过 `manage.py` 脚本可以方便地进行项目的管理和运行，包括数据库迁移、启动开发服务器等操作。项目的配置信息存储在 `settings.py` 文件中，包括数据库设置、静态文件路径、中间件配置等。