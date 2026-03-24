---
layout: post
title: "博客开发指南 (五)：打造高度定制化的备考数据大屏"
subtitle: "结合 Jekyll 静态数据与 Chart.js 的前端交互可视化"
date: 2026-03-24 10:00:00 +0800
author: "skyfury"
header-img: "img/post-bg-universe.jpg"
catalog: true
group: tech
tags:
  - JavaScript
  - Chart.js
  - Frontend
  - Data Visualization
---

在 `gwy.html` 页面中，我们将 Jekyll 纯静态博客的潜力推向了极致，构建了一个对齐 GitHub Contributions 风格的动态数据监控大屏。

## 静态博客如何做动态渲染？

Jekyll 是在构建时（Build Time）生成 HTML 的，没有运行时的数据库。那我们怎么计算本周的各科目学习时长并画出图表呢？

秘诀在于：**用 Liquid 在构建时将数据“注入”到 JavaScript 变量中**。

```javascript
const rawLogs = [
    {% for log in site.gwy_logs %}
    {
        date: "{{ log.date | date: '%Y-%m-%d' }}",
        totalHours: {{ log.study_hours | default: 0 }}, 
        subjects: [
            {% for sub in log.subjects %}
            { name: "{{ sub.name }}", hours: {{ sub.hours }} },
            {% endfor %}
        ]
    },
    {% endfor %}
];
```
这样，当用户访问网页时，客户端就会拿到一个纯净的 JSON 数组。接着由客户端 JS 进行聚合、过滤（区分本周和历史总计）、归一化（将“图推”映射为“判断推理”）。

## 交互式月历与时间轴

- **热力图月历**：我们手写了一个 Calendar Grid，根据日期匹配学习小时数，通过分配 CSS 类 `.day-level-1` 到 `.day-level-4`，呈现出深浅不同的绿色，非常直观地展现坚持打卡的情况。过去未打卡的日期还会触发红色 `.day-alert` 警示。
- **自定义卡片的时间轴**：周报与普通日志通过 Liquid 的 `if` 条件区分。如果是带有 `is_report` 标记的 AI 周报，渲染为金色带展开/折叠 (`<details>`) 功能的特殊卡片；若是普通日志，依据时长挂载不同层级的绿色左边框。

所有的 UI 调整都在 `_layouts/gwy_layout.html` 内通过精细的 CSS 和字体调整实现，不仅在 PC 端有左右布局的 Dashboard 体验，在移动端也能完美折叠适配。
        