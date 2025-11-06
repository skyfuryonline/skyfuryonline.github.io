# 如何在本博客中添加一篇新文章

本文档将详细说明如何在您的 Jekyll 博客中手动添加一篇新的文章，并将其正确地归入一个分组（新的或旧的），以及如何管理与之相关的图片。

---

## 核心概念

您的博客内容管理主要围绕以下几个目录和文件：

-   `_posts/`: 存放所有博客文章的 Markdown 文件。
-   `_groups/`: 存放每个分组的“着陆页”文件。
-   `_data/homepage_groups.yml`: 一个关键的配置文件，用于定义首页上展示的分组卡片，以及每个分组包含哪些文章。
-   `img/`: 存放所有静态图片资源的地方，包括文章的头图和文内插图。

---

## 场景一：向一个【已有分组】添加新文章

假设您想向已有的“博客开发指南”分组添加一篇新文章。

### 步骤 1: 添加文章的配图 (可选)

-   **文章头图 (Header Image)**: 您的文章布局 (`_layouts/post.html`) 设计得非常智能。它会自动查找该文章所属分组的背景图，并将其设为文章的头图。因此，对于加入已有分组的文章，您**通常不需要**单独为其指定头图。
-   **文章内插图**: 如果您的文章内容中需要插入图片，请将图片文件（例如 `my-new-image.png`）放入 `img/` 目录下的任意位置（建议按分组或文章名创建子目录以保持整洁，例如 `img/blog-development/`）。

### 步骤 2: 创建 Markdown 文章文件

1.  在 `_posts/` 目录下创建一个新的 Markdown 文件。
2.  **文件名必须严格遵循 Jekyll 的格式**: `YYYY-MM-DD-your-article-title.md`。
    -   例如: `2025-11-08-my-new-feature.md`。

### 步骤 3: 编写文章内容与 Front Matter

在新创建的文件中，您需要编写文章的 Front Matter (文件头部的配置) 和正文内容。

**Front Matter 是关键**，它告诉 Jekyll 如何处理这篇文章。一个最简化的配置如下：

```yaml
---
layout: post
title: "我的新功能"
date: 2025-11-08
author: "Your Name"
tags: [Jekyll, 新功能]  # 为文章打上标签
group: blog-development # 必须与 _groups 目录下的文件名对应
---

## 这是文章的标题

这是文章的正文内容。

![图片描述](/img/blog-development/my-new-image.png) # 引用您上传的文内插图
```

-   `layout: post`: 必须是 `post`。
-   `group: blog-development`: **这是最重要的一步**。这个值必须是您希望文章归属的分组的“标识符”（通常是 `_groups` 目录下对应的文件名，不含 `.md` 后缀）。

### 步骤 4: 更新分组配置文件 (手动)

这是当前架构下一个需要手动操作的步骤。

1.  打开 `_data/homepage_groups.yml` 文件。
2.  找到对应的分组（例如“博客开发指南”）。
3.  在 `posts:` 列表下，手动添加您刚刚创建的文章的**文件名**。

```yaml
- group_name: "博客开发指南"
  ...
  posts:
    - "2025-10-30-helloworld.md"
    - ...
    - "2025-11-08-my-new-feature.md" # <-- 在这里添加新行
```

**为什么需要这一步？** 因为文章的头图是根据这篇文章属于哪个分组来决定的。系统通过遍历这个 `posts` 列表，来确定文章与分组的从属关系，从而找到正确的头图 `group_image`。

---

## 场景二：创建一个【全新分组】并添加文章

假设您想创建一个名为“生活随想”的新分组。

### 步骤 1: 为新分组准备背景图

1.  选择一张您喜欢的图片作为新分组的背景图和文章默认头图。
2.  将它放入 `img/` 目录，例如 `img/cover-life.png`。

### 步骤 2: 创建分组的“着陆页”

1.  在 `_groups/` 目录下创建一个新的 Markdown 文件。
2.  **文件名将作为该分组的唯一标识符**。例如，`life-thoughts.md`。
3.  文件内容非常简单，只需指定布局和标题即可：
    ```yaml
    ---
    layout: group_page
    title: "生活随想"
    permalink: /groups/life-thoughts # 定义该分组页面的 URL
    ---
    ```

### 步骤 3: 在首页上展示新分组

1.  打开 `_data/homepage_groups.yml` 文件。
2.  在文件末尾添加一个完整的新分组配置节：

    ```yaml
    - group_name: "生活随想"
      url: "/groups/life-thoughts" # 必须与上面 permalink 对应
      group_image: "/img/cover-life.png" # 指向您上传的背景图
      group_description: "记录生活中的点滴感悟。"
      posts:
        [] # 暂时为空，等待添加文章
    ```

### 步骤 4: 向这个新分组添加文章

现在，这个新分组已经创建完毕。您可以完全按照【场景一】中的步骤，来向这个新分组添加文章了。唯一的区别是，在文章的 Front Matter 中，`group` 字段的值，以及在 `homepage_groups.yml` 中 `posts` 列表的更新，都将指向您新创建的分组标识符 `life-thoughts`。
