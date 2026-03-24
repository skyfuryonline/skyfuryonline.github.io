# 基于 Jekyll、GitHub Actions 与 LLM 的现代个人知识库

本项目是一个高度定制化的 Jekyll 静态博客，核心特色是采用了 **代码与内容分离的双仓库架构**，并深度集成了 **Python 自动化工作流** 和 **具备长时记忆的 AI 学习教练**。

![页面展示](img/screenshot.png)

## 架构特色

1. **内容与源码分离 (双仓库设计)**：
   - **源码仓库 (当前仓库)**：负责存放 Jekyll 布局、CSS 样式、前端交互脚本（Chart.js 大屏）以及 Python 自动化部署脚本。
   - **数据仓库 (`blog-source`，私有)**：负责存放所有包含个人隐私的 Markdown 文章、日记和 AI 生成的周报数据。
   - **安全性**：通过 GitHub Actions，在云端 CI/CD 流程中拉取私有数据进行静态渲染。博客源码即使公开，也绝对不会泄漏未发布的草稿或个人敏感日记。

2. **数据驱动的模块化设计**：
   - 首页专栏完全由 `_data/homepage_groups.yml` 驱动（分为 Tech, Life, Vision, Gwy 等）。
   - 任何专栏和页面的增减无需修改 HTML，只需修改配置文件和补充对应的 Markdown 路由即可。

3. **动态可交互的备考数据大屏 (Gwy)**：
   - 使用 Chart.js 与自定义的月历热力图，构建了类似 GitHub Contributions 风格的仪表盘。
   - 可以在纯静态页面上，动态统计和展示每日/每周的学习时长、连续打卡天数以及各科目的精力分布。

4. **AI 原生学习教练**：
   - 在后端的自动化流程中集成了基于 LLM 的生成器 (`crawlers/weekly_summary_generator.py`)。
   - **长时递归记忆**：AI 不仅总结本周的日志，还会自动读取上周/上月的旧报告，实现状态和问题的跨周继承，给出连贯的教练建议。
   - LLM 的配置（模型切换、系统提示词）被统一抽离在 `crawlers/config.json` 中，可轻松无缝切换不同厂商大模型。

## 部署工作流 (.github/workflows/deploy.yml)

整个发布流程完全自动化：
1. **触发**: 推送代码或内容仓库有更新时，触发 Actions。
2. **拉取依赖**: 拉取源码仓库及私有的内容仓库（使用 PAT Token）。
3. **内容融合**: 将私有仓库中的文章和日志利用 `rsync` 挂载到 Jekyll 的对应目录。
4. **自动化脚本**: 运行 Python 环境，执行 AI 周报/月报自动生成或外部爬虫获取。
5. **构建部署**: 调用 Jekyll 插件进行静态编译，发布至 GitHub Pages。

## 极速创作指南

本仓库内置了 VS Code 的专属工作流设置，极大优化了写作体验。

**如何写新文章：**
1. 在 `_posts/` 或内容库对应目录下新建文件，格式如 `YYYY-MM-DD-your-title.md`。
2. 输入 **`!post`** 并按 `Tab` 或 `Enter` 键。
3. 系统将自动生成包含今天日期的完整 Jekyll Front Matter。
4. 使用 `Tab` 键即可在**标题**、**文章分组**（下拉菜单选择如 tech, life, vision）和**标签**中快速跳转补全，无需手敲繁琐的 YAML。
5. 如需写学习日记，可以使用 **`!gwy`** 命令，同样一键补全。

## 目录结构速览

- `.github/workflows/deploy.yml`: 自动化部署编排。
- `_layouts/`, `_includes/`: 页面排版（如 `gwy_layout.html` 自定义了大屏组件）。
- `_data/`: 驱动首页分类和卡片的数据源。
- `crawlers/`: Python 自动化脚本集。
  - `weekly_summary_generator.py`: 带长时记忆的 AI 报告生成引擎。
  - `config.json`: LLM 调度与爬虫配置文件。
- `.vscode/`: 包含快捷输入指令 `markdown.code-snippets` 及环境配置。