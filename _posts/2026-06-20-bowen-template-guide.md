---
layout: post
title: "博闻笔记模板指南"
subtitle: "文章排版组件、公式支持与使用方法"
date: 2026-06-20
author: "LH"
catalog: true
tags:
  - "Workflow"
keywords:
  - "模板"
  - "排版"
  - "KaTeX"
  - "博闻"
group: bowen
---

<link rel="stylesheet" href="/css/bowen.css">

这篇文章是「博闻」分组的文章排版模板参考，同时也是数学公式渲染的测试页。

## Front Matter

每篇文章的 YAML 头部应包含以下字段：

```yaml
---
layout: post
title: "文章标题"
subtitle: "一句话概括核心内容"
date: 2026-xx-xx
author: "LH"
catalog: true
tags:
  - "分类标签"
keywords:
  - "关键词"
group: bowen
---
```

正文第一行引入样式表：

```html
<link rel="stylesheet" href="/css/bowen.css">
```

同时将文章文件名添加到 `_data/homepage_groups.yml` 中「博闻」分组的 `posts` 列表。

## 来源信息卡片

<div class="bowen-source">
  <div class="bowen-source-label">📺 视频来源</div>
  <div class="bowen-source-title">视频标题 / 文章标题</div>
  <div class="bowen-source-meta">
    <span>UP主 / 作者名称</span>
    <span>时长: 12:34</span>
    <span><a href="https://bilibili.com/video/BVxxxxxx" target="_blank">B站链接</a></span>
  </div>
</div>

> `label` 可根据来源类型替换：📺 视频来源、📖 文章来源、🎓 课程来源

<hr class="bowen-divider">

## 数学公式支持

博客已集成 KaTeX 渲染引擎，支持 LaTeX 语法公式。

**行内公式**：用单美元符号包裹，如 `$E = mc^2$` 渲染为 $E = mc^2$，或者 $\nabla \cdot \mathbf{E} = \frac{\rho}{\varepsilon_0}$。

**块级公式**：用双美元符号包裹：

$$
\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}
$$

$$
P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}
$$

**矩阵与多行公式**：

$$
\mathbf{A} = \begin{pmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{pmatrix}
$$

> **注意**：kramdown 会把行内的 `$...$` 原样保留给 KaTeX 处理。如果公式内有特殊字符导致渲染异常，可以用 `\(...\)` 代替 `$...$`，用 `\[...\]` 代替 `$$...$$`。

<hr class="bowen-divider">

## 关键词胶囊

正文中用 <span class="bowen-kw">关键词胶囊</span> 标注重要术语。

HTML: `<span class="bowen-kw">关键词</span>`

<hr class="bowen-divider">

## 要点高亮框

<div class="bowen-highlight">
  <strong>💡 核心要点</strong>
  <p>需要重点掌握的内容：</p>
  <ul>
    <li>第一个要点——简明扼要</li>
    <li>第二个要点——附带解释</li>
    <li>第三个要点——关键结论</li>
  </ul>
</div>

> 标题文字可自定义：💡 核心要点、🔑 关键结论、📝 学习笔记、⚠️ 易错点

<hr class="bowen-divider">

## 词汇网格（语言学习）

<div class="bowen-vocab">
  <div class="bowen-vocab-item">
    <span class="bowen-vocab-word">ephemeral</span>
    <span class="bowen-vocab-phonetic">/ɪˈfem.ər.əl/</span>
    <span class="bowen-vocab-mean">短暂的，转瞬即逝的</span>
  </div>
  <div class="bowen-vocab-item">
    <span class="bowen-vocab-word">ubiquitous</span>
    <span class="bowen-vocab-phonetic">/juːˈbɪk.wɪ.təs/</span>
    <span class="bowen-vocab-mean">无处不在的，普遍存在的</span>
  </div>
  <div class="bowen-vocab-item">
    <span class="bowen-vocab-word">pragmatic</span>
    <span class="bowen-vocab-phonetic">/præɡˈmæt.ɪk/</span>
    <span class="bowen-vocab-mean">务实的，实用主义的</span>
  </div>
  <div class="bowen-vocab-item">
    <span class="bowen-vocab-word">serendipity</span>
    <span class="bowen-vocab-phonetic">/ˌser.ənˈdɪp.ə.ti/</span>
    <span class="bowen-vocab-mean">意外发现美好事物的运气</span>
  </div>
</div>

> 音标行 `bowen-vocab-phonetic` 是可选的，非语言类笔记可省略。

<hr class="bowen-divider">

## 编号洞察卡片

<div class="bowen-insight">
  <span class="bowen-insight-num">1</span>
  <div class="bowen-insight-body">
    <strong>第一个关键发现</strong>
    <p>详细说明，包含具体分析和思考。公式也可内嵌：$\sum_{i=1}^{n} i = \frac{n(n+1)}{2}$</p>
  </div>
</div>

<div class="bowen-insight">
  <span class="bowen-insight-num">2</span>
  <div class="bowen-insight-body">
    <strong>第二个关键发现</strong>
    <p>卡片自动编号，视觉层次清晰。</p>
  </div>
</div>

<hr class="bowen-divider">

## 章节引言

<p class="bowen-intro">章节引言样式，用在大章节开头概括本节内容。使用 <code>bowen-intro</code> 类。</p>

## 完整文章结构

1. `<link>` — 引入样式
2. **来源卡片**（`.bowen-source`）— 标注出处
3. **正文** — 自由组织，`$公式$` 直接用 LaTeX，`.bowen-kw` 标注术语
4. **高亮框**（`.bowen-highlight`）— 穿插要点
5. **词汇网格**（`.bowen-vocab`）— 语言类专用
6. **洞察卡片**（`.bowen-insight`）— 文末总结

所有组件均可选，按内容灵活选用。
