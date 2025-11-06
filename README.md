# 基于 Jekyll、GitHub Action、LLMs的个人博客搭建

本项目是一个基于 Jekyll 的博客，其核心特色是集成了一套复杂的、自动化的内容聚合系统。该系统利用由 GitHub Actions 编排的 Python 爬虫，来自动抓取、总结并展示来自多个高质量技术博客的内容。
![页面展示](img/screenshot.png)

## 核心功能

- **自动化内容抓取**: 每日定时执行 Python 爬虫，以发现并抓取最新的文章。
- **动态内容处理**: 利用 Selenium 来渲染重度依赖 JavaScript 的网站，确保在解析前内容已完全加载。
- **AI 驱动的摘要**: 集成大语言模型 (LLM)，为每篇抓取的文章生成精炼的摘要。
- **图片与媒体缓存**: 自动下载并缓存文章中的图片，使其可被 Jekyll 站点直接访问。
- **CI/CD 编排**: 由 GitHub Actions 管理的全自动化工作流，负责抓取、数据处理和最终部署。
- **清晰的架构**: 在爬虫（负责数据抓取）和主编排器（负责数据处理、缓存和 LLM 交互）之间实现了明确的关注点分离。

## 工作原理

整个流程由定义在 `.github/workflows/deploy.yml` 中的 GitHub Actions 工作流进行编排。

1.  **触发**: 工作流会按每日计划、手动触发，或在每次推送到 `master` 分支时运行。
2.  **编排 (`crawlers/main.py`)**:
    - 初始化一个共享的 Selenium WebDriver 实例，供所有爬虫使用。
    - 从 `crawlers/config.json` 读取配置，以确定要抓取哪些网站。
    - 遍历所有启用的爬虫。
3.  **抓取 (`crawlers/specific_crawlers/`)**:
    - 每个具体的爬虫都负责从其目标站点抓取一个文章列表。
    - 对于每篇新文章，它会抓取其完整的文章文本和所有图片的 URL。
    - 然后，它将这些结构化数据（文本和图片 URL）返回给主编排器。
4.  **处理与缓存 (`crawlers/main.py`)**:
    - 对于接收到的每篇文章，编排器会创建一个经过净化的缓存目录（例如 `cache/YYYY-MM-DD/article-title/`）。
    - 它将文章文本保存到 `content.txt` 中。
    - 它会异步地将所有图片从提供的 URL 下载到该缓存目录中。
    - 它调用 LLM 摘要器 (`llm/summarizer.py`) 来生成内容摘要。
5.  **数据生成**:
    - 编排器将当天所有文章的元数据（标题、链接、来源、摘要、相对缓存路径、图片文件名）编译到一个 JSON 文件中（例如 `_data/daily_YYYY-MM-DD.json`）。
6.  **Jekyll 构建与部署**:
    - GitHub Actions 工作流接着会将新生成的数据文件提交到 `data` 分支。
    - 最后，它触发 Jekyll 的构建和部署流程，Jekyll 会使用 `_data` 目录中的数据来渲染 `daily.html` 页面。

## 项目结构

-   `.github/workflows/deploy.yml`: 主要的 GitHub Actions 工作流文件。
-   `crawlers/`: 内容聚合系统的核心。
    -   `main.py`: 主编排器脚本。
    -   `config.json`: 用于所有爬虫、站点和 LLM 提示词的配置文件。
    -   `specific_crawlers/`: 包含针对每个目标站点的具体爬虫实现。
-   `llm/`: 包含与 LLM 交互以进行摘要的逻辑。
-   `_data/`: 最终生成的、供 Jekyll 使用的每日 JSON 文件存放处。
-   `cache/`: 所有下载的内容（文本和图片）的缓存位置。
-   `daily.html`: 读取 `_data` 中的数据并展示聚合内容的 Jekyll 页面。

## 依赖项

所有 Python 相关的依赖项都在 `requirements.txt` 文件中列出。
