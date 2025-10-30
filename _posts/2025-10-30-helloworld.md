---
layout: post
title:  "如何发布新博客及管理图片"
subtitle: "一份关于本博客维护的简易指南"
date:   2025-10-30 16:00:00 +0800
author: "LH"
tags:
  - 博客维护
  - 教程
---

将操作指南本身作为第一篇博客，既实用又有纪念意义。

### 第一步：创建并上传博客文章

1.  **创建 Markdown 文件**:
    *   在您项目的 `_posts` 文件夹中，创建一个新的 `.md` 文件。
    *   **文件名必须严格遵守 `YYYY-MM-DD-your-post-title.md` 的格式**。例如：`2025-11-01-my-first-post.md`。

2.  **编写文章头部信息 (Front Matter)**:
    *   在每个 Markdown 文件的最上方，您需要添加一个 YAML front matter 块。这是 Jekyll 用来识别文章属性的部分。
    *   您可以复制并修改下面的模板：

    ```yaml
    ---
    layout: post
    title:  "我的第一篇博客"
    subtitle: "这是一个副标题，可以留空"
    date:   2025-11-01 10:00:00 +0800
    author: "LH"
    header-img: "img/my-first-post/header.jpg"  # 这篇文章的头部背景图片
    tags:
      - 学习笔记
      - NLP
    ---
    ```

    *   `title`: 文章主标题。
    *   `subtitle`: 副标题（可选）。
    *   `date`: 文章发布日期。
    *   `header-img`: **这篇文章独立的背景图片**。如果留空或删除这行，它会使用您在 `_config.yml` 中设置的全局背景图。
    *   `tags`: 文章的标签，方便分类。

3.  **编写博客正文**:
    *   在头部信息下方，直接用 Markdown 语法编写您的博客内容即可。

### 第二步：管理并引用文章图片

为每篇文章创建一个专属的图片文件夹是最佳实践。

1.  **创建图片文件夹**:
    *   在项目的 `img` 文件夹内，创建一个与您博客文章标题对应的子文件夹。
    *   例如，对于文章 `2025-11-01-my-first-post.md`，您可以创建一个名为 `my-first-post` 的文件夹。完整路径为 `img/my-first-post/`。

2.  **存放图片**:
    *   将这篇文章需要用到的所有图片（包括头部背景图 `header.jpg` 和文章内图片 `image1.png` 等）都放入这个新建的文件夹中。

3.  **在文章中引用图片**:
    *   在您的 Markdown 正文或 `header-img` 中，使用**以 `/` 开头的绝对路径**来引用图片。这样做可以确保无论在哪一页（主页、文章页），图片都能正确显示。
    *   **引用头部背景图 (Front Matter 中)**:
        ```yaml
        header-img: "/img/my-first-post/header.jpg"
        ```
    *   **引用文章内图片 (Markdown 正文)**:
        ```markdown
        ![这是一张示例图片](/img/my-first-post/image1.png)
        ```

### 第三步：发布博客

完成以上步骤后，只需将您的修改提交到 GitHub 即可。

1.  打开终端，进入您的项目目录 (`D:\self-built-web-proj\skyfuryonline.github.io`)。
2.  执行以下命令：

    ```shell
    git add .
    git commit -m "feat: Add new post 'My First Post'"
    git push
    ```

几分钟后，您的新博客文章就会出现在网站上了。