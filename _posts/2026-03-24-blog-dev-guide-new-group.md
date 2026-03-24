---
layout: post
title: "博客开发指南 (五)：数据驱动的新分组创建"
subtitle: "不用写 HTML，教你一分钟在首页增加新专栏"
date: 2026-03-24 10:00:00 +0800
author: "skyfury"
header-img: "img/cover-zaji.png"
catalog: true
tags:
  - Jekyll
  - Liquid
  - Data-Driven
---

博客运营一段时间后，你一定会有拓展新领域的需求。比如你原来只写技术，现在想增加一个叫“音乐分享”的新专栏。
在传统的博客系统里，你可能需要去修改 HTML，写新的 `div`，调 CSS 布局。但在这个博客里，一切都被设计成了**数据驱动**！

## 三步新建专栏

假设我们要新建一个叫“音乐分享”的分类：

### 第一步：注册分组数据
打开 `_data/homepage_groups.yml`，在文件末尾加上这一段：

```yaml
- group_name: "音乐分享"
  url: "/groups/music"
  group_image: "/img/cover-music.svg"
  group_description: "分享最近听到的好歌与感受"
  posts: []
```

### 第二步：配置封面图
将你的封面图片（最好是设计感强的 SVG 或高质量图片）放进 `img/` 目录下，并命名为 `cover-music.svg`，对应上面 `group_image` 填写的路径。

### 第三步：挂载文章路由
为了让点击卡片后能跳到一个真正的页面，我们需要在 `_groups/` 目录下新建一个文件 `music.md`，并在里面填入几行配置：
```yaml
---
layout: page
title: "音乐分享"
description: "分享最近听到的好歌与感受"
header-img: "img/cover-music.svg"
---

{% raw %}{% include group_posts.html group_name="音乐分享" %}{% endraw %}
```

**搞定了！**
这就是全部的过程。你甚至连一行 HTML 和 CSS 都没碰过，Jekyll 的 Liquid 模板会在下一次构建时，自动在首页渲染出这张崭新且响应式居中的卡片。

以后你要在这个专栏里发文章，只需要在写文章时，把那篇文章的文件名添加到 `_data/homepage_groups.yml` 对应分组的 `posts` 列表中即可！
        