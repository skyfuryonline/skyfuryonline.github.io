---
layout: post
title: "为Jekyll博客集成一个全自动的每日信息爬虫（保姆级教程）"
subtitle: "从Python脚本到GitHub Actions，实现动态抓取、缓存与前端展示"
date: 2025-11-03 18:00:00
author: "LH"
tags: [GitHub, Jekyll, Python, Crawler, CI/CD]
group: life
---

## 前言：让静态博客“活”起来

Jekyll 是一个非常出色的静态网站生成器，但“静态”二字有时也意味着内容的更新依赖于手动的 `git push`。如果我们想让博客的某个页面能每天自动更新，展示来自其他网站的最新信息（比如新闻、技术文章、论文列表等），该怎么做呢？

这篇博文将是一份保姆级的实战教程，详细记录我们如何从零开始，为本博客集成一个全自动的每日信息爬虫系统。我们将实现以下目标：

1.  **自动化**: 利用 GitHub Actions 实现定时任务，无需人工干预。
2.  **模块化**: 构建一个可扩展的 Python 爬虫框架，方便未来添加更多爬取源。
3.  **智能化**: 具备查重、自动清理过期数据等功能。
4.  **动态展示**: 将爬取到的数据动态渲染到博客的 "Daily" 页面。

让我们开始吧！

## 第一阶段：规划与基础框架搭建

凡事预则立，不预则废。一个清晰的计划是成功的一半。我们首先规划了整个项目的架构和流程。

### 1. 最终目标

我们希望在博客的导航栏增加一个 "Daily" 页面，该页面每天会自动更新，展示我们从特定网站（以博客园精选为例）抓取到的最新文章列表。

### 2. 技术选型

*   **自动化**: GitHub Actions
*   **爬虫框架**: Python + `crawl4ai` (一个对LLM友好的爬虫库)
*   **数据存储**: 抓取到的内容缓存到 `cache/` 目录，元数据保存到 Jekyll 能直接读取的 `_data/` 目录。

### 3. 核心设计

我们确定了几个核心的设计原则：

*   **按日期缓存**: 每天爬取的内容存放在以当天日期命名的文件夹中（如 `cache/2025-11-03/`），结构清晰。
*   **数据去重**: 每次爬取前，先加载最近15天已爬取的文章URL，避免重复抓取。
*   **自动清理**: 每次任务结束时，自动删除超过15天的旧缓存和数据，防止仓库无限膨胀。
*   **配置驱动**: 要爬取的目标网站和解析器名称，都定义在 `config.json` 中，方便扩展。
*   **面向对象**: 使用“基类+子类”的模式，将通用逻辑（如文件保存）和特定网站的解析逻辑解耦。

这是我们当时绘制的蓝图，也是接下来所有步骤的依据：

```markdown
# 爬虫与博客集成计划 (v2)

**分支策略：**
1.  `feature/crawler-foundation`: 搭建爬虫基础框架。
2.  `feature/llm-integration`: 集成 LLM API 进行总结。

---

### 第一阶段：爬虫基础框架

1.  **环境准备**: 确定 Python 版本和依赖库版本。
2.  **搭建目录结构**: 创建 `crawlers/` 和 `llm/` 文件夹。
3.  **实现爬虫框架**: 编写 `base_crawler.py`, `main.py`, `config.json` 和具体的爬虫子类。
4.  **集成到 GitHub Actions**: 修改 `.github/workflows/deploy.yml`，加入 Python 环境和爬虫运行步骤。

### 第二阶段：LLM 集成与数据展示

...
```

## 第二阶段：编码实现 - 打造爬虫核心

现在，我们进入激动人心的编码环节。我们将一步步创建出爬虫的各个模块。

### 1. 环境准备 (`requirements.txt`)

首先，我们需要一个 `requirements.txt` 文件来管理我们的Python依赖。固定版本号是一个非常好的习惯，它可以保证在任何环境下（包括云端的GitHub Actions）安装的都是相同版本的库，避免因库更新导致意外的错误。

```txt
# requirements.txt

crawl4ai==0.7.6
requests==2.32.5
beautifulsoup4==4.12.2
lxml==5.4.0 # crawl4ai 0.7.6 需要 lxml 5.3 以上版本
jsonlines==3.1.0
aiohttp==3.9.5 # 异步HTTP请求
```

### 2. 爬虫基类 (`crawlers/base_crawler.py`)

我们首先创建一个 `base_crawler.py` 文件。这个基类的作用是封装所有爬虫都通用的逻辑，比如保存文本内容、异步下载图片等。子类只需要继承它，然后专注于如何解析特定网站即可。

```python
# crawlers/base_crawler.py

from abc import ABC, abstractmethod
import os
import json
import aiohttp

class BaseCrawler(ABC):
    """所有爬虫的抽象基类。"""

    def __init__(self, url, output_dir):
        self.url = url
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    async def save_content(self, article_dir, content, images):
        """将文本内容和图片保存到本地目录。"""
        os.makedirs(article_dir, exist_ok=True)

        # 1. 保存文本内容
        with open(os.path.join(article_dir, "content.txt"), "w", encoding="utf-8") as f:
            f.write(content)

        # 2. 异步下载并保存图片
        async with aiohttp.ClientSession() as session:
            for i, img_url in enumerate(images):
                # 确保图片URL是绝对路径
                if not img_url.startswith('http'):
                    img_url = f"https:{img_url}" if img_url.startswith('//') else f"{self.url.rsplit('/', 1)[0]}/{img_url}"
                
                img_path = os.path.join(article_dir, f"image_{i+1}.jpg")
                try:
                    async with session.get(img_url, timeout=10) as response:
                        if response.status == 200:
                            with open(img_path, "wb") as img_file:
                                img_file.write(await response.read())
                        else:
                            print(f"Warning: Failed to download image {img_url} with status {response.status}")
                except Exception as e:
                    print(f"Error: Failed to download image {img_url}: {e}")

    def save_metadata(self, metadata_path, items):
        """将元数据保存为JSON文件。"""
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=4)

    @abstractmethod
    async def crawl(self):
        """爬取网站并返回结构化数据。这是子类必须实现的方法。"""
        pass
```

### 3. 博客园爬虫 (`crawlers/specific_crawlers/cnblogs_crawler.py`)

接下来，我们创建第一个具体的爬虫，用于抓取博客园精选文章。它继承自 `BaseCrawler`，并实现了具体的 `crawl` 方法。

```python
# crawlers/specific_crawlers/cnblogs_crawler.py

from crawl4ai import AsyncWebCrawler
from bs4 import BeautifulSoup
import os
import re
import time
import random
from ..base_crawler import BaseCrawler

def sanitize_filename(filename):
    """移除在Windows/Linux文件名中的非法字符。"""
    return re.sub(r'[\\/*?"<>|:]', '-', filename)

class CnblogsCrawler(BaseCrawler):
    """博客园爬虫。"""

    def __init__(self, url: str, output_dir: str, existing_urls: set, top_k: int = 5):
        super().__init__(url, output_dir)
        self.top_k = top_k
        self.existing_urls = existing_urls # 接收来自main.py的历史URL集合

    async def crawl(self):
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36"}
        crawler = AsyncWebCrawler(headers=headers)
        
        # 1. 爬取精选文章列表页
        list_page_result = await crawler.arun(url=self.url)
        if not list_page_result.success:
            print(f"Failed to crawl list page {self.url}: {list_page_result.error_message}")
            return []

        soup = BeautifulSoup(list_page_result.html, "lxml")
        article_links = soup.select("a.post-item-title")[:self.top_k]

        metadata_items = []

        # 2. 遍历并处理每一篇文章
        for tag in article_links:
            title = tag.get_text(strip=True)
            link = tag.get("href", "")
            if not link.startswith('http'):
                link = f"https://www.cnblogs.com{link}"

            # 查重：如果文章URL已存在，则跳过
            if link in self.existing_urls:
                print(f"Skipping already processed article: {title}")
                continue

            print(f"Processing new article: {title}")

            # 规范化文件名
            safe_title = sanitize_filename(title)
            article_dir = os.path.join(self.output_dir, safe_title)

            time.sleep(random.uniform(1, 2)) # 礼貌地等待一下

            # 爬取文章详情页
            article_result = await crawler.arun(url=link)
            if not article_result.success:
                print(f"Failed to crawl article {link}: {article_result.error_message}")
                continue

            # 使用crawl4ai获取Markdown正文
            content = article_result.markdown
            
            # 使用BeautifulSoup在正文区域内精确提取图片
            article_soup = BeautifulSoup(article_result.html, 'lxml')
            content_body = article_soup.find('div', id='cnblogs_post_body')
            images = []
            if content_body:
                images = [img['src'] for img in content_body.find_all('img') if img.get('src')]
            
            # 调用基类的保存方法
            await self.save_content(article_dir, content, images)

            # 准备元数据
            metadata_items.append({
                "title": title,
                "link": link,
                "cache_path": os.path.join(self.output_dir, safe_title)
            })

        return metadata_items
```

### 4. 配置文件 (`crawlers/config.json`)

这是一个简单的JSON文件，用来告诉主程序要运行哪些爬虫。

```json
{
    "sites": [
        {
            "url": "https://www.cnblogs.com/pick/",
            "parser": "CnblogsCrawler"
        }
    ]
}
```

### 5. 总编排脚本 (`crawlers/main.py`)

这是我们爬虫系统的大脑，负责所有流程的编排：加载历史、运行爬虫、保存当天数据、清理过期数据。

```python
# crawlers/main.py

import json
import importlib
import os
import asyncio
import shutil
import re
from datetime import datetime, timedelta
from pathlib import Path

# ... (camel_to_snake, load_existing_urls, cleanup_old_data 函数代码) ...

async def main():
    """爬虫总编排脚本"""
    print("Starting crawler orchestration...")
    
    # 1. 设置路径并加载历史URL
    today = datetime.now().strftime("%Y-%m-%d")
    # ... (路径设置代码) ...
    existing_urls = load_existing_urls(data_dir, days_to_keep=15)
    print(f"Found {len(existing_urls)} existing URLs from the last 15 days.")

    # 2. 加载配置
    # ... (加载 config.json 代码) ...

    # 3. 动态运行所有爬虫
    all_metadata = []
    for site in config["sites"]:
        # ... (动态导入并运行爬虫的代码) ...
        # 注入依赖：当天的缓存目录和历史URL集合
        crawler_instance = CrawlerClass(site["url"], todays_cache_dir, existing_urls)
        metadata = await crawler_instance.crawl()
        all_metadata.extend(metadata)

    # 4. 保存当天的元数据到 _data 目录
    if all_metadata:
        with open(todays_data_file, 'w', encoding='utf-8') as f:
            json.dump(all_metadata, f, ensure_ascii=False, indent=4)
        print(f"Successfully saved today's metadata to {todays_data_file}")
    else:
        print("No new articles found to save.")

    # 5. 清理15天前的旧数据
    cleanup_old_data(cache_dir, data_dir, days_to_keep=15)

# ... (主函数入口代码) ...
## 第四阶段：调试与总结

在我们将所有代码合并到 `master` 分支后，我们遇到了一个有趣的现象：**代码推送后，"Daily" 页面上并没有立刻出现内容，并且在 GitHub Actions 的日志中，“Run crawler” 步骤的图标是一个灰色的斜杠 (/)，而不是绿色的勾 (✓)。**

![](/img/crawler-log-skipped.png) *(这里可以放一张GitHub Actions日志截图)*

### “跳过”的斜杠 (/) 是怎么回事？

这其实是**完全符合我们预期**的正常现象！

在 GitHub Actions 的日志中，不同图标的含义是：
*   **绿色的勾 (✓)**: 步骤**成功运行**。
*   **红色的叉 (✗)**: 步骤**运行失败**。
*   **灰色的斜杠 (/)**: 步骤被**跳过 (Skipped)**。

我们之前在 `deploy.yml` 中为爬虫步骤设置了 `if` 条件，导致它只在定时或手动触发时运行。由于我们是通过 `git push` 触发的工作流，`github.event_name` 的值是 `push`，不满足条件，所以 Actions **正确地跳过了**这个步骤。自然，页面上也就不会有新数据了。

### 如何看到最终效果？

为了真正运行爬虫并看到效果，我们需要手动触发一次工作流：
1.  打开 GitHub 仓库页面，点击 **"Actions"** 标签。
2.  在左侧选择我们的工作流（例如 "Jekyll site CI"）。
3.  点击 **"Run workflow"** 下拉按钮，再次确认运行。

手动触发后，我们就能在日志中看到 “Run crawler” 步骤被成功执行（绿色的勾 ✓），稍等片刻，访问我们的博客，"Daily" 页面上就会出现从博客园抓取到的最新文章列表了！

### 总结

通过这次实践，我们成功地将一个动态的、自动化的爬虫系统，无缝集成到了一个静态的 Jekyll 博客中。我们学习了如何：

*   使用 Python 构建一个模块化、可扩展的爬虫框架。
*   利用 GitHub Actions 的 `schedule` 和 `if` 条件，实现精确的自动化任务控制。
*   解决在 CI/CD 环境中由 `Playwright` 等复杂工具链带来的依赖问题。
*   通过 Jekyll 的 `_data` 目录和 `Liquid` 模板语言，将动态数据渲染到静态页面上。

现在，我们的博客拥有了一个能自我更新的“信息聚合器”，真正地“活”了起来。希望这篇详细的教程能对你有所帮助！

---

### **附录：又一个调试案例——定时任务为什么没在“早上8点”运行？**

在我们将所有功能都部署好之后，第二天早上，我们发现 "Daily" 页面并没有更新。我们检查了 Actions 的运行记录，发现并没有在早上8点由 `schedule` 触发的记录。

**问题根源：时区！**

这是 CI/CD 中一个极其常见的陷阱。我们在 `deploy.yml` 中写的 `cron: '0 8 * * *'`，直观地理解是“每天早上8点”。但问题是，这是哪个时区的早上8点？

**GitHub Actions 的所有 `schedule` 事件，都严格基于 UTC (协调世界时) 时间。**

*   我们期望的是 **北京时间 (CST)** 的早上 8 点。
*   北京时间比 UTC 时间快 8 个小时 (UTC+8)。
*   因此，当北京时间是早上 8:00 时，UTC 时间其实是 0:00。

所以，我们设置的 `0 8 * * *` (UTC 8:00)，实际上会在北京时间下午 16:00 (下午4点) 才会运行。

**解决方案：**

将 `cron` 表达式调整为正确的 UTC 时间即可。

```yaml
  schedule:
    - cron: '0 0 * * *' # 对应 UTC 0:00，即北京时间 8:00
```

这个小小的案例提醒我们，在处理任何与“时间”相关的自动化任务时，永远要第一时间确认其基准时区，这能为我们省下大量的调试时间。

---

### **附录B：最终优化——为什么修改了UI，刷新后却看不到？**

在我们完成了所有功能，并对 "Daily" 页面的 UI 进行了精细的调整后，我们遇到了最后一个，也是最经典的一个前端问题：

**“我明明已经把最新的代码部署上去了，为什么刷新页面后，看到的还是旧的样式？”**

即使用户强制刷新（`Ctrl+Shift+R`），有时也无法立刻看到最新的布局。这对于用户体验来说是不可接受的。

**问题根源：浏览器缓存 (Browser Caching)**

为了提高网站的加载速度，浏览器会把已经下载过的静态资源（比如 CSS 样式表和 JS 脚本）保存在本地。当用户再次访问时，浏览器会直接使用这些本地的旧文件，而不是去服务器请求最新的版本。这就是为什么我们部署了新的 CSS，但浏览器依然在使用旧的样式。

**解决方案：缓存破坏 (Cache Busting)**

我们需要一种方法来“欺骗”浏览器，让它认为这是一个全新的文件，从而主动去下载。最简单、最有效的实现方式是：**为静态资源的 URL 添加一个每次构建都会改变的查询字符串。**

我们在 `_includes/head.html` 中找到了引用主样式表的地方，并做了如下修改：

**修改前:**
```html
<link rel="stylesheet" href="{{ "/css/hux-blog.min.css" | prepend: site.baseurl }}">
```

**修改后:**
```html
<link rel="stylesheet" href="{{ "/css/hux-blog.min.css" | prepend: site.baseurl }}?v={{ site.time | date: '%s' }}">
```

**这是如何工作的？**
*   `site.time` 是 Jekyll 在**构建网站时**的当前时间。
*   `date: '%s'` 过滤器将这个时间格式化为 **Unix 时间戳**（一个独一无二的、不断增长的秒数）。
*   最终生成的 URL 会像这样：`/css/hux-blog.min.css?v=1762041600`。

每次我们的网站通过 GitHub Actions 重新构建时，都会生成一个新的时间戳。浏览器看到这个新的 URL，就会认为这是一个它从未见过的新文件，从而放弃旧的缓存，下载最新的版本。

我们对所有主要的 CSS 和 JS 文件都应用了同样的操作。这样一来，就彻底解决了因浏览器缓存导致的用户无法看到最新更新的问题，完成了我们从后端数据到前端体验的最后一公里。

## 第三阶段：自动化与展示

代码已经准备就绪，现在我们需要让它在云端自动运行，并将结果展示在我们的博客上。

### 1. 集成到 GitHub Actions (`.github/workflows/deploy.yml`)

我们的目标是让爬虫只在**定时任务**或**手动触发**时运行，而在普通的 `git push` 时跳过，以节省资源。我们通过修改 `deploy.yml` 工作流文件来实现这一点。

```yaml
# .github/workflows/deploy.yml

# ... (省略了 on, permissions, concurrency 等设置) ...

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout 🛎️
        uses: actions/checkout@v4

      # 新增：设置 Python 3.11 环境
      - name: Setup Python 🐍
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      # 新增：安装 Python 依赖
      - name: Install Python dependencies 📦
        run: pip install -r requirements.txt

      # 关键！新增：安装 Playwright 浏览器内核
      - name: Install Playwright Browsers 🎭
        run: playwright install

      # 新增：运行爬虫脚本，并设置触发条件
      - name: Run crawler 🕷️
        if: github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'
        run: python crawlers/main.py

      - name: Setup Ruby and install gems 💎
        # ... (原有的 Jekyll 设置步骤)

      - name: Build the site 🏗️
        # ... (原有的 Jekyll 构建步骤)

      - name: Upload artifact 📦
        # ... (原有的上传步骤)

# ... (省略了 deploy job)
```

**最重要的修改有三处：**
1.  **安装 Python 和依赖**: 使用 `actions/setup-python` 并运行 `pip install`。
2.  **安装 Playwright 浏览器**: 这是我们在调试中发现的关键一步。`crawl4ai` 底层使用 `Playwright`，它需要一个真实的浏览器内核才能工作。`playwright install` 命令会自动下载并安装它。
3.  **条件化运行爬虫**: `if: github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'` 这行代码确保了爬虫只在我们想要它运行的时候运行。

### 2. 前端页面展示 (`daily.html`)

最后一步，我们需要修改 `daily.html` 页面，让它能读取 `_data/` 目录下的数据并渲染出来。我们使用 Jekyll 的 `Liquid` 模板语言来实现。

```html
--- 
layout: page
title: "Daily"
---

<style>
.post-card { /* ... 省略样式 ... */ }
</style>

<div class="container">
    <div class="row">
        <div class="col-lg-8 col-lg-offset-2 col-md-10 col-md-offset-1">
            
            {% comment %} 1. 找到最新的 daily_*.json 文件 {% endcomment %}
            {% assign latest_date = "1970-01-01" %}
            {% for file in site.data %}
                {% if file[0] contains "daily_" %}
                    {% assign file_date_str = file[0] | remove: "daily_" %}
                    {% if file_date_str > latest_date %}
                        {% assign latest_date = file_date_str %}
                    {% endif %}
                {% endif %}
            {% endfor %}

            {% comment %} 2. 读取并展示最新文件中的文章 {% endcomment %}
            {% if latest_date != "1970-01-01" %}
                {% assign latest_data_key = "daily_" | append: latest_date %}
                {% assign articles = site.data[latest_data_key] %}

                <p class="post-meta">Showing articles from: {{ latest_date }}</p>

                {% for article in articles %}
                    <a href="{{ article.link }}" target="_blank" class="post-card">
                        <h2>{{ article.title }}</h2>
                        <p class="post-meta">Source: cnblogs</p>
                    </a>
                {% endfor %}
            {% else %}
                <p>No daily information available yet. Please run the crawler.</p>
            {% endif %}

        </div>
    </div>
</div>
```
这段代码的逻辑很清晰：首先遍历 `site.data` 找到最新的日期，然后加载对应的数据文件，最后通过一个 `for` 循环将每篇文章渲染成一个可点击的卡片。
