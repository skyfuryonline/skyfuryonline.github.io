# 如何为本系统添加一个新的网站爬虫

本文档将详细说明向本自动化博客内容聚合系统中添加一个新的网站爬虫所需的完整步骤。得益于系统良好、解耦的架构，您只需编写少量代码并修改一个配置文件即可完成扩展。

---

## 核心流程

添加一个新爬虫，您只需要做 **2** 件事：

1.  在 `crawlers/specific_crawlers/` 目录下**创建一个新的爬虫 Python 文件**，并实现其核心抓取逻辑。
2.  在 `crawlers/config.json` 中**添加一个对应的配置节**，来启用和配置您的新爬虫。

您完全**不需要**修改 `main.py` 或任何其他核心逻辑，系统会自动发现并运行您的新爬虫。

---

## 第一步：编写新的爬虫类

这是主要的工作。您需要在 `crawlers/specific_crawlers/` 目录下，创建一个新的 Python 文件。

### 1. 文件命名规范

文件名必须是您打算创建的爬虫类名的 “蛇形命名法” (snake_case) 版本。这是因为主编排器 `main.py` 会根据类名自动推断并加载对应的文件名。

-   **示例**：如果您的新爬虫类叫做 `MyAwesomeBlogCrawler`，那么文件名必须是 `my_awesome_blog_crawler.py`。

### 2. 类的基本结构

在这个新文件中，您需要定义您的爬虫类。它必须遵循以下规则：

-   **继承基类**：必须继承自 `BaseCrawler`。
-   **初始化方法**：必须有一个 `__init__` 方法，接收 `url`, `cache_dir`, `existing_urls`, `driver`, 和 `top_k` 作为参数，并调用 `super().__init__(...)`。
-   **实现 `crawl` 方法**：必须实现一个名为 `crawl` 的异步方法 (`async def crawl(self):`)。

### 3. `crawl` 方法的职责

`crawl` 方法是您编写核心抓取逻辑的地方。您需要使用 `self.driver` (这是一个共享的 Selenium 实例) 来访问网页、查找元素、提取数据。

该方法最终**必须返回一个列表**，列表中的每个元素都是一个字典，代表一篇文章。每个字典**必须包含**以下五个 `key`：

-   `title`: 文章标题 (字符串)
-   `link`: 文章的永久链接 (字符串)
-   `date`: 文章日期 (字符串, 格式为 `YYYY-MM-DD`)
-   `content`: 文章的纯文本内容 (字符串)
-   `image_urls`: 一个包含该文章所有图片**完整 URL** 的列表 (字符串列表)

### 4. 代码框架示例

以下是一个可供您参考和复制的框架，文件名为 `my_awesome_blog_crawler.py`：

```python
# 引入所有你需要的库，如 time, BeautifulSoup, urljoin 等
from crawlers.base_crawler import BaseCrawler
from datetime import datetime

class MyAwesomeBlogCrawler(BaseCrawler):
    def __init__(self, url, cache_dir, existing_urls, driver, top_k=5):
        super().__init__(url, cache_dir, existing_urls, driver)
        self.top_k = top_k

    async def crawl(self):
        articles_to_return = []
        
        # --- 在这里编写您的抓取逻辑 ---
        # 1. 使用 self.driver.get(self.url) 访问列表页
        # 2. 找到所有文章的链接和标题
        # 3. 遍历链接，并使用 `if link in self.existing_urls: continue` 来去重
        # 4. 对于新文章，访问其详情页，抓取 content 和 image_urls
        # 5. 将抓取到的数据组装成一个字典，append 到 articles_to_return 列表中
        
        # 示例：
        new_article = {
            'title': "一篇很棒的文章",
            'link': "https://example.com/awesome-post",
            'date': datetime.now().strftime("%Y-%m-%d"),
            'content': "这是文章内容...",
            'image_urls': ["https://example.com/image1.jpg", "https://example.com/image2.png"]
        }
        articles_to_return.append(new_article)
        
        return articles_to_return
```

---

## 第二步：更新配置文件

这是最简单的一步。您只需要打开 `crawlers/config.json` 文件，然后在 `sites` 数组中，添加一个新的 JSON 对象来配置您的新爬虫。

**示例**:
```json
{
  "global_settings": { ... },
  "sites": [
    {
      "parser": "GoogleDevBlogCrawler",
      ...
    },
    // ... 其他已有的爬虫
    {
      "parser": "MyAwesomeBlogCrawler", // 必须与您创建的类名完全一致
      "url": "https://awesome-blog.com/archive", // 您要抓取的入口页面
      "top_k": 5, // 您希望从这个网站最多抓取几篇文章
      "llm_profile": "default_summary", // 使用哪个 LLM 配置来做摘要
      "enabled": true // 设置为 true 来启用它
    }
  ],
  "llm_profiles": { ... }
}
```

完成以上两步后，当您下一次运行爬虫系统时，您的新爬虫就会被自动加载并执行。
