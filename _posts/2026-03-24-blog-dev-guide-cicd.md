---
layout: post
title: "博客开发指南 (二)：GitHub Actions 自动化部署流"
subtitle: "解密网站背后的自动化构建与发布流水线"
date: 2026-03-24 10:00:00 +0800
author: "LH"
group: blog-development
catalog: true
tags:
  - GitHub Actions
  - CI/CD
  - 自动化
---

作为现代化的技术博客，我们不再依赖手动在本地跑命令生成页面，而是将这一切都交给了 **GitHub Actions**。

## CI/CD 核心工作流

在 `.github/workflows/deploy.yml` 文件中，我们定义了博客的完整构建流程。这套流程会在两种情况下被触发：
1. **代码推送**：只要我们向 `master` 分支推送了新的 Markdown 文章或代码修改，流程就会立即启动。
2. **定时触发 / 手动触发**：由于我们有每日爬虫，每天也会按时启动一次构建。

## 部署流程图解

每当构建启动时，GitHub 云端服务器（Actions Runner）会按顺序执行以下任务：

1. **Checkout (拉取代码)**：
   不仅会拉取当前仓库的公开源码，还会利用 PAT（Personal Access Token）安全地拉取私有的内容仓库，将文章数据无缝融入进来。

2. **安装 Python 环境与依赖**：
   博客中的爬虫和数据处理需要 Python。这一步会自动执行 `pip install -r requirements.txt`，确保 `Selenium`、`BeautifulSoup` 以及 `Requests` 等爬虫依赖库就绪。

3. **执行每日抓取任务**：
   如果是定时触发，系统会执行我们写的 Python 爬虫脚本，去各个技术网站抓取最新的内容并生成 JSON 数据。

4. **Jekyll 构建**：
   使用官方的 `jekyll-build` 插件，将所有的 Markdown、HTML 模板和 CSS 样式，结合刚刚爬取到的数据，编译出最终的纯静态 HTML 文件集合。

5. **部署到 GitHub Pages**：
   最后一步，将编译好的产物直接发布到 GitHub Pages 的高速 CDN 节点上。

这套流水线的好处在于：**完全的“免运维”**。在本地，你只需要安心写 Markdown，用 Git Push 提交后，几分钟内你的新文章就会在全球网络中更新！
        