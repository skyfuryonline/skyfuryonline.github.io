---
layout: post
title: "博客爬虫(续)：从AI总结到完美UI与部署"
subtitle: "实现图文模态框，并解决依赖冲突、权限、数据持久化等一系列真实世界问题"
date: 2025-11-04 10:00:00
author: "LH"
tags: [LLM, UI, Jekyll, CI/CD, Debugging]
---

## 前言：从“是什么”到“讲什么”，再到“看什么”

在[上一篇教程](2025-11-03-add-crawler-to-jekyll-blog.md)中，我们为博客集成了一个全自动的爬虫。但这仅仅是开始。一个好的产品，不仅要有内容，更要有优秀的体验。

这篇教程，我们将进行一次彻底的“体验升级”。我们将实现以下目标：

1.  **集成 LLM**: 实现对抓取文章的**自动摘要**功能。
2.  **升级 UI**: 设计一个图文并茂的**模态框 (Modal)**，在不离开页面的情况下，为用户提供“摘要 + 图片”的快速预览。
3.  **解决部署难题**: 记录并解决我们在集成过程中遇到的**所有真实世界的部署问题**，包括依赖冲突、权限错误、数据持久化策略等。这部分将是本教程最有价值的地方。

## 第一部分：LLM 集成与 UI 设计

### 1. 设计哲学：预生成与数据驱动

我们的核心依然是“预生成”。所有需要计算和API调用的步骤，都在构建时完成。最终，前端拿到的 `daily_*.json` 文件，将包含渲染一个完美模态框所需的所有信息：

```json
{
    "title": "文章标题",
    "link": "原文链接",
    "summary": "AI总结...",
    "cache_path": "cache/2025-11-04/文章标题",
    "image_files": ["image_1.jpg", "image_2.jpg"]
}
```

### 2. 后端改造：获取图片列表与相对路径

为了实现图文展示，我们对爬虫脚本做了两处关键修改：

*   **获取图片列表**: 在 `main.py` 中，我们增加了扫描缓存目录、获取所有图片文件名的逻辑。
*   **修正路径问题**: 这是我们遇到的第一个大坑。最初，我们错误地将服务器的绝对文件路径存入了 `cache_path`。我们必须修正 `cnblogs_crawler.py`，让它存入相对于网站根目录的**相对路径**，例如 `cache/2025-11-04/...`。

```python
# crawlers/cnblogs_crawler.py (关键修改)

# ...
# NOTE: We must store a relative path for the web URL, not the absolute filesystem path.
relative_cache_path = os.path.join("cache", os.path.basename(self.output_dir), safe_title)
metadata_items.append({
    "title": title,
    "link": link,
    "cache_path": relative_cache_path.replace('\\', '/') # Ensure forward slashes
})
```

### 3. 前端实现：模态框、图片画廊与 URL 编码

我们在 `daily.html` 中进行了大量的前端工作：

*   **HTML**: 使用 Bootstrap 的标准模态框结构，并为文章卡片添加 `data-*` 属性来存储摘要、图片列表和缓存路径。
*   **CSS**: 编写了图片画廊的样式。为了让图片美观且不变形，我们最终采用了 `object-fit: cover` 方案，它能让不同尺寸的图片都完美地填充一个固定大小的容器。

    ```css
    .gallery-item {
        height: 200px; /* 给容器一个固定高度 */
        /* ... 其他 flex 布局属性 ... */
    }
    .gallery-item img {
        width: 100%;
        height: 100%; /* 让图片填满容器 */
        object-fit: cover; /* 裁剪以适应，不变形 */
    }
    ```

*   **JavaScript**: 这是交互的核心。当卡片被点击时：
    1.  从 `data-*` 属性中读取所有数据。
    2.  将标题和摘要填充到模态框中。
    3.  **URL 编码**：这是我们遇到的第二个大坑。由于文章标题是中文，直接拼接的 URL 是非法的。我们必须使用 `encodeURIComponent` 对路径的每个部分进行编码，才能生成合法的图片 `src`。

    ```javascript
    // daily.html (关键修改)

    // ...
    var pathParts = (cachePath + '/' + imageFile).split('/').map(function(part) {
        return encodeURIComponent(part);
    });
    img.src = '{{ site.baseurl }}/' + pathParts.join('/');
    ```

## 第二部分：部署的“九九八十一难”

当我们把这一切推送到 GitHub Actions 时，真正的挑战才刚刚开始。我们几乎遇到了所有新手在做 CI/CD 时会遇到的经典问题。

### 难关一：Python 依赖冲突

*   **问题**: `crawl4ai` 需要 `anyio>=4.0.0`，而旧版的 `openai` 库需要 `anyio<4.0.0`，两者完全冲突，导致 `pip install` 失败。
*   **解决方案**: 在 `requirements.txt` 中，**不锁定 `openai` 的版本**，只写 `openai`。这等于授权 `pip` 的依赖解析器，让它自己去寻找一个能与 `crawl4ai` 和谐共存的、最新的 `openai` 版本。

### 难关二：Python 缩进错误

*   **问题**: `IndentationError`。一个低级但致命的错误，因为在修改代码时破坏了 Python 的缩进结构。
*   **解决方案**: 仔细检查报错的行号，修正缩进。这是每个 Python 程序员的必经之路。

### 难关三：数据持久化的权限问题

*   **问题**: 我们希望 Actions 将爬取的数据提交回仓库，但 `git push` 报了 `403 Permission Denied` 错误。
*   **原因**: Actions 的默认令牌 `GITHUB_TOKEN` 没有向仓库写入的权限。
*   **解决方案**: 在 `deploy.yml` 中，为 `build` 任务明确授予 `contents: write` 权限。

    ```yaml
    jobs:
      build:
        permissions:
          contents: write # 授予写入权限
        steps:
        # ...
    ```

### 难关四：代码与数据的“战争”

*   **问题**: 我们本地在修改代码，而 Actions 在自动提交数据，当我们 `git pull` 时，会因为数据文件太多而卡死，或者产生大量合并冲突。
*   **最终解决方案：代码与数据分离**
    1.  **`master` 分支**: 只保存代码。`.gitignore` 中忽略 `cache/` 和 `_data/daily_*.json`。
    2.  **`data` 分支**: 创建一个全新的、孤立的分支，专门用于存放 `cache/` 和 `_data/`。
    3.  **改造工作流**: `deploy.yml` 被彻底改造。它会同时 `checkout` 两个分支，构建时将两者结合，构建结束后，**只将新数据推送回 `data` 分支**。`master` 分支永远不会再被 Actions 自动修改。

    ```yaml
    # deploy.yml (核心逻辑)
    steps:
      - name: Checkout master branch (code) 🛎️
        uses: actions/checkout@v4
        with:
          path: main

      - name: Checkout data branch (data) 📦
        uses: actions/checkout@v4
        with:
          ref: data
          path: data
      
      # ... (运行爬虫，在 main 目录生成数据)

      - name: Commit and push data to data branch 💾
        run: |
          # 1. 把 main 目录里新生成的数据，复制到 data 目录
          cp -rf main/_data/. data/_data/
          cp -rf main/cache/. data/cache/

          # 2. 进入 data 目录，执行提交
          cd data
          git add -A
          git commit -m "chore: Update daily data"
          git push
    ```

### 难关五：`.gitignore` 的“两面性”

*   **问题**: 即使我们采用了双分支策略，`git push` 依然报错，提示 `_data` 目录被 `.gitignore` 忽略了。
*   **原因**: Actions 在 `data` 目录中执行 `git add` 时，依然受到了 `master` 分支检出到 `main` 目录中的那个 `.gitignore` 文件的影响。
*   **解决方案**: 在 `deploy.yml` 的提交步骤中，使用 `git add -A` (或 `-f`) 来强制添加文件，无视 `.gitignore` 的规则。

### 第九难：数学公式渲染失败之谜

**问题描述**:
在我撰写一篇关于 `Minimind-v` 的技术笔记时，我希望在文章中插入一个 LaTeX 公式，例如 `$\frac{1}{7000}$`。然而，在页面上，它并没有被渲染成漂亮的分数，而是被原封不动地显示为纯文本字符串。

**排查过程与最终诊断**:
这个问题远比想象的要复杂，我经历了一场漫长而曲折的排查：

1.  **怀疑 Markdown 引擎**: 我首先怀疑 Jekyll 的 Markdown 引擎（Kramdown）没有正确处理数学公式。但在检查 `_config.yml` 后，发现 `markdown: kramdown` 的设置是完全正确的。
2.  **怀疑 MathJax 未加载**: 接着，我怀疑页面可能根本没有加载用于渲染数学公式的 MathJax 库。于是，我尝试通过 CDN 在 `<head>` 中显式加载 MathJax。**失败**。
3.  **怀疑 CDN 问题**: 考虑到网络因素，我将 MathJax 库下载到了本地，并修改 `<head>` 中的脚本，使其从本地 (`/js/mathjax.js`) 加载。**依然失败**。
4.  **怀疑脚本加载时序**: 此时，问题指向了脚本的加载时序。在某些主题中，放置在 `<head>` 的脚本可能会在页面 DOM 完全构建之前执行，导致找不到需要渲染的元素。于是，我将 MathJax 脚本从 `<head>` 移至 `<footer>` 的末尾，确保它在所有 HTML 内容都加载完毕后才运行。

**根本原因分析**:
经过上述所有步骤后，问题依然存在，这使我最终确认了问题的两个核心根源：

*   **Markdown 转义冲突 (首要原因)**: Markdown 解释器会先于 MathJax 运行。它会将 LaTeX 命令中的反斜杠 `\` 视为转义字符并“吃掉”。例如，`\frac` 会被处理成 `frac`，导致 MathJax 无法识别这个残缺的指令。
*   **脚本加载的健壮性 (次要原因)**: 虽然我的主题已经包含了 MathJax，但它的加载方式可能不够稳健，或者存在潜在的冲突。显式地、在正确的位置（footer）加载一个我们自己控制的 MathJax 库版本，是确保渲染引擎正常工作的前提。

**最终解决方案**:
一个“双保险”方案，彻底解决了这个顽固的问题：

1.  **内容层面——二次转义**: 在 Markdown 文件中，对所有 LaTeX 命令的反斜杠 `\`，都进行二次转义，即 `\` -> `\\`。例如，将 `$\frac{1}{7000}$` 修改为 `$\\frac{1}{7000}$`。
2.  **架构层面——本地化并后置脚本**:
    *   将 MathJax 库下载到本地的 `js/` 目录下。
    *   在 `_includes/footer.html` 的末尾，添加加载和配置本地 MathJax 脚本的代码，确保它在页面完全加载后才执行。

这个过程告诉我们，在复杂的 Web 渲染环境中，问题往往是多因素叠加的结果。只有系统性地排查从后端（Jekyll/Kramdown）到前端（HTML结构/JS加载时序/内容转义）的每一个环节，才能最终定位并解决问题。

## 总结：没有一帆风顺的工程

从一个简单的想法，到一个功能完善、体验良好、部署健壮的自动化系统，我们几乎把新手能踩的坑都踩了一遍。但正是这个过程，让我们对 CI/CD、Git 多分支协作、前端路径处理、CSS 布局等一系列工程问题，有了无比深刻和具象的理解。

现在，我们的博客终于拥有了一个能真正让人愉悦地使用的、自我更新的“情报中心”。这个过程虽然曲折，但结果令人满意。
