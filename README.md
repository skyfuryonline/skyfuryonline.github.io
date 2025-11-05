# LH的博客

![screenshot](img/screenshot.png)

这是一个基于 Jekyll 和 GitHub Pages 搭建的个人博客，用于记录博主在NLP领域的学习、研究以及生活点滴。

## 核心特性

- **自动化内容聚合**: 每日自动通过 GitHub Actions 运行 Python 爬虫，抓取外部网站的最新文章。
- **AI 自动摘要**: 利用大语言模型（LLM）API，为每篇抓取的文章自动生成摘要。
- **现代化 UI/UX**: 在 "DAILY" 页面通过模态框，以图文并茂的形式展示摘要，提升阅读体验。
- **并发性能优化**: 通过 `asyncio` 实现对 LLM API 的并发调用，极大提升多篇文章的摘要生成效率。
- **代码与数据分离**: 采用双分支策略，将网站代码 (`master` 分支) 与爬虫数据 (`data` 分支) 完全分离，确保了开发的轻量与部署的健壮。

## 网站结构

本项目的架构经过精心设计，以实现自动化和可维护性：

- **`master` 分支 (代码)**
  - `_config.yml`: 网站的全局配置文件。
  - `_posts/`: 存放手动撰写的博客文章。
  - `_layouts/`, `_includes/`: Jekyll 布局与组件。
  - `daily.html`, `about.html`, `tags.html`: 各个主要页面。
  - `assets/`, `css/`, `js/`, `img/`: 网站的静态资源。
  - `.github/workflows/deploy.yml`: **核心 CI/CD 工作流**。负责定时触发、检出代码和数据、运行爬虫、构建和部署网站，并将新数据推送回 `data` 分支。
  - `crawlers/`: **Python 爬虫系统**
    - `main.py`: 主编排脚本，负责调度、查重、并发调用 LLM 和清理过期数据。
    - `config.json`: 爬虫的配置文件，用于定义爬取目标、`top_k` 策略、LLM 配置等。
    - `base_crawler.py`: 所有爬虫的基类，封装了通用方法。
    - `specific_crawlers/`: 存放针对特定网站的爬虫实现。
  - `llm/`: **LLM 摘要模块**
    - `summarizer.py`: 封装了对 OpenAI 兼容 API 的**异步**调用，以实现高效并发的摘要生成。
  - `.gitignore`: 在 `master` 分支上，明确忽略 `cache/` 和 `_data/daily_*.json`，以保持代码库的纯净。

- **`data` 分支 (数据)**
  - `_data/`: 存放由爬虫生成的 `daily_YYYY-MM-DD.json` 文件，供 Jekyll 在构建时使用。
  - `cache/`: 存放爬取到的文章原文 (`content.txt`) 和图片，作为生成摘要和前端展示的数据源。

## 参考

本博客的主题修改自以下开源项目，感谢原作者的分享：

- **[qiubaiying/qiubaiying.github.io](https://github.com/qiubaiying/qiubaiying.github.io)**

## 如何在本地运行

1.  **安装 Ruby 和 Jekyll:**
    请参考 Jekyll 官方文档：[https://jekyllrb.com/docs/installation/](https://jekyllrb.com/docs/installation/)

2.  **安装依赖:**
    在项目根目录下运行 `bundle install`。

3.  **启动本地服务:**
    运行 `bundle exec jekyll serve`，然后通过浏览器访问 `http://localhost:4000`。
