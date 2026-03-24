---
layout: post
title: "博客开发指南 (一)：项目架构与核心页面解析"
subtitle: "从 0 认识本博客的文件结构与页面构成"
date: 2026-03-24 10:00:00 +0800
author: "skyfury"
header-img: "img/post-bg-universe.jpg"
catalog: true
tags:
  - Jekyll
  - Architecture
---

欢迎来到本博客的开发指南系列！这套指南将带你深入了解我们是如何构建和维护这个现代化静态博客的。首篇文章，我们先来剖析项目的整体结构以及最重要的几个公共页面。

## 项目目录结构

我们的博客基于 Jekyll 构建，整个项目的目录树非常清晰：

```text
├── _data/                 # 数据源目录（决定了首页的分组和各类配置）
├── _includes/             # 页面可复用的组件（如页眉、页脚）
├── _layouts/              # 页面的基础骨架模板
├── _posts/                # 你正在阅读的每一篇博客文章
├── _groups/               # 存放不同专栏/分组的路由页面
├── crawlers/              # Python 自动化爬虫脚本（Daily页面的动力源）
├── img/                   # 全站静态图片与 SVG 资源
├── .github/workflows/     # GitHub Actions CI/CD 的定义文件
├── .vscode/               # VS Code 的快捷指令配置
├── index.html             # 博客的首页 (Home)
└── daily.html             # 基于爬虫生成的资讯聚合页
```

## 核心页面解析

### 1. 首页 (Home - `index.html`)

进入博客，你看到的第一眼就是 `index.html`。
它并未像传统博客那样直接平铺显示所有文章，而是采用了**专栏卡片式**的设计。首页会读取 `_data/homepage_groups.yml` 里的数据，通过循环遍历渲染出不同的学习专栏（如“模型及分布式训练笔记”、“视野”等）。用户点击对应的卡片，才会进入对应的文章列表页。

### 2. 每日资讯聚合页 (Daily - `daily.html`)

这是一个非常特殊的页面，它让静态博客拥有了“动态”的能力。
`daily.html` 专门用来展示每天全自动抓取回来的高质量技术文章和论文。它的内容并不是我们手写的，而是由后端的 Python 爬虫和 LLM 生成 JSON 数据注入而成的。关于它是如何运作的，我们将在后续的“爬虫与 LLM”篇章中详细讲解。

接下来的一篇，我们将看看这套静态代码是如何被推送到云端并自动发布的！
        