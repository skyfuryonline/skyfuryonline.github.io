---
layout: post
title: "博客开发指南 (一)：现代 Jekyll 架构与内容代码分离"
subtitle: "利用 GitHub Actions 将私有内容库动态注入公共展示页，兼顾隐私与分享"
date: 2026-03-24 10:00:00 +0800
author: "skyfury"
header-img: "img/post-bg-universe.jpg"
catalog: true
group: tech
tags:
  - Jekyll
  - GitHub Actions
  - CI/CD
  - Architecture
---

本文将介绍如何从 0 搭建这套现代化的 Jekyll 博客，并着重讲解“内容与源码分离”的架构设计。

## 为什么需要内容与源码分离？

许多静态博客（如 Hexo, Hugo, 传统 Jekyll）将 Markdown 文章和网站的 HTML/CSS 源码放在同一个仓库。但这会带来一个痛点：**如果我有些学习日记或草稿不想公开，但又想利用这套系统进行本地管理，怎么办？**

本站通过 **双仓库架构** 彻底解决了这个问题：
1. **公开源码仓库 (`skyfuryonline.github.io`)**：存放所有的 UI 组件、Jekyll 配置文件、CSS 样式、爬虫脚本以及 Actions 部署流程。
2. **私有内容仓库 (`blog-source`)**：专门存放核心的 Markdown 帖子、备考日记（`_gwy_logs`）、每周 AI 生成报告（`_gwy_reports`）等含有个人隐私的数据。

## GitHub Actions：连接两端的桥梁

每次我们需要发布新内容时，只需要把 Markdown 文件 push 到 `blog-source`，然后触发 `skyfuryonline.github.io` 仓库的 GitHub Action。

核心工作流 (`.github/workflows/deploy.yml`) 如下：
1. **Checkout 公开仓库**。
2. **Checkout 私有仓库**：使用 Personal Access Token (PAT) 赋予权限，将 `blog-source` 的内容拉取到当前目录。
3. **内容合并**：通过 `rsync` 命令，将私有仓库中的文章、日记复制到 Jekyll 对应的 `_posts/`, `_gwy_logs/` 目录。
4. **依赖安装与脚本执行**：安装 Python 依赖（通过 `requirements.txt` 统一管理），运行自动化脚本（如周报生成、爬虫等）。
5. **Jekyll 构建与部署**：使用原生的 Jekyll 插件构建静态 HTML，并部署到 GitHub Pages。

## 流程优势
- **极度安全**：哪怕公开仓库的页面被完全爬取，那些配置了 `published: false` 或者不放在 `_posts` 里的隐私文件也绝不会暴露在源码中。
- **环境隔离**：任何复杂的环境依赖（如 Playwright、LiteLLM 或 BeautifulSoup）仅在 CI/CD 中通过 Python 虚拟环境执行，不污染本地开发机器。
        