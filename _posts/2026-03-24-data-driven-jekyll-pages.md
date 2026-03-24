---
layout: post
title: "博客开发指南 (三)：数据驱动的设计与首页多分组"
subtitle: "利用 YAML 数据文件与 Jekyll Collections 构建高扩展性前端"
date: 2026-03-24 10:00:00 +0800
author: "skyfury"
header-img: "img/post-bg-universe.jpg"
catalog: true
group: tech
tags:
  - Jekyll
  - Liquid
  - YAML
  - Frontend
---

这个博客不仅仅是一个“文章列表”，而是一个具备多个领域的知识库（技术、生活、视野）。为了避免在各个 HTML 模板里写死分类名称，我们采用了**完全数据驱动**的设计。

## 核心设计：`_data` 目录

Jekyll 提供了一个强大的功能：Data Files。我们可以把配置写在 `_data/homepage_groups.yml` 里：

```yaml
- id: tech
  name: 极客
  desc: Coding
  cover: "img/cover-tech.jpg"

- id: vision
  name: 视野
  desc: Emerging Tech & Trends
  cover: "img/cover-vision.svg"
```

## Liquid 模板：动态渲染首页

在首页模板中，我们不用手写一个个卡片，而是遍历这份数据文件：
```liquid
{% for group in site.data.homepage_groups %}
  <div class="group-card">
    <img src="{{ group.cover }}">
    <h3>{{ group.name }}</h3>
    <p>{{ group.desc }}</p>
  </div>
{% endfor %}
```
这种设计的好处在于：**增加新板块时，完全不需要修改前端 HTML 或 CSS！**
只要在 YAML 里加一项，画一张 `cover-vision.svg`，然后在 `_groups/` 目录下新建一个路由文件（如 `vision.md` 绑定该 id），一个全新的板块就自动上线了。

## CSS 的模块化配合

配合这种数据驱动模式，我们在 CSS 中广泛使用了 Flexbox 和 CSS Grid 布局，确保卡片数量变化时，无论是 2 个、3 个还是 4 个，页面都能自适应居中或对齐，不会出现布局崩塌。
        