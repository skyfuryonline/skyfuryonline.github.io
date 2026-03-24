---
layout: post
title: "博客开发指南 (三)：构建基于 LLM 的每日资讯爬虫"
subtitle: "探索 Daily 页面背后的大模型自动摘要引擎"
date: 2026-03-24 10:00:00 +0800
author: "LH"
group: blog-development
catalog: true
tags:
  - Crawler
  - LLM
  - Python
---

如果你访问过博客的 `Daily` 页面，你会发现那里总有最新鲜的技术资讯。这都归功于我们在 `crawlers/` 目录下编写的一套自动化爬虫与 LLM 摘要引擎。

## 系统是如何工作的？

每天，GitHub Actions 都会唤醒我们的爬虫系统。整个过程分为三个明确的阶段：

### 1. 动态抓取阶段
在 `crawlers/specific_crawlers/` 目录下，我们针对不同的信息源（如各大技术社区）编写了对应的解析逻辑。
为了应对现代网页中大量依靠 JavaScript 渲染的内容，我们集成了 **Selenium** 无头浏览器。它可以模拟真实人类打开网页的动作，等网页完全加载完毕后，再由 `BeautifulSoup` 接手提取正文文本和图片的 URL。

### 2. 媒体缓存阶段
获取到正文后，主调度脚本 (`crawlers/main.py`) 会在后台异步地将文章里提到的关键图片下载到本地的 `cache/` 目录下。这样哪怕原网站的图片链接失效或者被防盗链拦截，我们的博客依然能正常展示图文并茂的信息。

### 3. LLM 智能摘要
把成篇的几万字直接推给读者体验并不好，所以我们引入了 **LLM（大语言模型）**。
系统会读取 `crawlers/config.json` 里的统一配置，提取我们设定的 `system_prompt`。然后通过对应的 Python SDK，将这篇长文的纯净文本发送给 AI 模型，让其生成一段 100 字左右的精炼总结。

### 4. 最终数据注入
所有抓取到的标题、链接、本地缓存图片路径以及 AI 生成的摘要，都会被汇编成一个巨大的 JSON 文件放入博客的 `_data/` 目录。
当 Jekyll 随后启动构建时，Liquid 语法就能轻松地读取这些 JSON 字典，瞬间将它们铺满了 `daily.html` 页面！
        