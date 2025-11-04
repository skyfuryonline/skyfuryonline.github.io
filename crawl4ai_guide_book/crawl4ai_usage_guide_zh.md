# Crawl4AI 完整使用说明文档

## 目录
1. [简介](#简介)
2. [安装](#安装)
3. [基本用法](#基本用法)
4. [高级特性](#高级特性)
5. [网站爬取技术](#网站爬取技术)
6. [与LLM集成](#与llm集成)
7. [API调用](#api调用)
8. [多网站爬取实现](#多网站爬取实现)
9. [最佳实践](#最佳实践)

## 简介

Crawl4AI是一个现代的、高性能的网络爬虫框架，专为与大型语言模型(LLM)集成而设计。它支持动态内容抓取、结构化数据提取、智能内容过滤、并行处理等高级功能，可以轻松处理JavaScript渲染的页面、无限滚动页面和需要登录的网站。

## 安装

### 基础安装
```bash
# 安装基础包
pip install -U crawl4ai

# 安装预发布版本
pip install crawl4ai --pre

# 运行安装后设置
crawl4ai-setup

# 验证安装
crawl4ai-doctor
```

如果遇到浏览器相关问题，手动安装：
```bash
python -m playwright install --with-deps chromium
```

### 带可选功能的安装
```bash
# 同步版本（已弃用）
pip install crawl4ai[sync]

# 包含PyTorch功能
pip install crawl4ai[torch]

# 包含Transformers功能
pip install crawl4ai[transformer]

# 包含余弦相似度功能
pip install crawl4ai[cosine]

# 安装所有可选功能
pip install crawl4ai[all]
```

### 开发版安装
```bash
git clone https://github.com/unclecode/crawl4ai.git
cd crawl4ai
pip install -e .                    # 可编辑模式的基础安装

# 安装可选功能：
pip install -e ".[torch]"           # 包含PyTorch功能
pip install -e ".[transformer]"     # 包含Transformer功能
pip install -e ".[cosine]"          # 包含余弦相似度功能
pip install -e ".[sync]"            # 包含同步爬取（Selenium）
pip install -e ".[all]"             # 安装所有可选功能
```

### Docker部署
```bash
# 拉取并运行最新版本
docker pull unclecode/crawl4ai:latest
docker run -d -p 11235:11235 --name crawl4ai --shm-size=1g unclecode/crawl4ai:latest

# 访问游乐场： http://localhost:11235/playground
```

## 基本用法

### Python中的简单网络爬取
```python
import asyncio
from crawl4ai import AsyncWebCrawler

async def main():
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url="https://www.nbcnews.com/business",
        )
        print(result.markdown)

if __name__ == "__main__":
    asyncio.run(main())
```

### 命令行界面使用
```bash
# 基础爬取，输出Markdown格式
crwl https://www.nbcnews.com/business -o markdown

# 深度爬取，使用BFS策略，最多爬取10页
crwl https://docs.crawl4ai.com --deep-crawl bfs --max-pages 10

# 使用LLM提取特定问题
crwl https://www.example.com/products -q "提取所有产品价格"
```

## 高级特性

### Markdown生成
- 清晰、结构化的Markdown，格式准确
- 启发式过滤，去除噪音和无关内容
- 带编号引用列表的引用和参考文献
- 针对性定制的Markdown生成策略
- 使用BM25算法提取核心信息

### 结构化数据提取
- 支持所有LLM（开源和专有）的LLM驱动提取
- 分块策略（基于主题、正则表达式、句子级别）
- 基于查询的余弦相似度查找相关内容
- 使用XPath和CSS选择器的CSS基础提取
- 自定义模式定义的结构化JSON提取

### 浏览器集成
- 全控制的浏览器管理
- 通过Chrome开发者工具协议的远程浏览器控制
- 保存认证状态的持久化浏览器配置
- 多步骤爬取的会话管理
- 带认证的代理支持
- 完整浏览器控制（头、cookie、用户代理）
- 多浏览器支持（Chromium、Firefox、WebKit）
- 动态视口调整

### 爬取和抓取功能
- 媒体提取（图像、音频、视频、响应式格式）
- 带JS执行的动态爬取
- 爬取过程中的页面截图
- 原始数据爬取（HTML、本地文件）
- 综合链接提取（内部、外部、iframe）
- 每个步骤的自定义钩子
- 改进性能的缓存
- 元数据提取
- IFrame内容提取
- 懒加载处理
- 无限滚动页面的全页扫描

## 网站爬取技术

### 启发式Markdown生成
```python
import asyncio
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.content_filter_strategy import PruningContentFilter, BM25ContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

async def main():
    browser_config = BrowserConfig(
        headless=True,  
        verbose=True,
    )
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.ENABLED,
        markdown_generator=DefaultMarkdownGenerator(
            content_filter=PruningContentFilter(
                threshold=0.48, 
                threshold_type="fixed", 
                min_word_threshold=0
            )
        ),
        # markdown_generator=DefaultMarkdownGenerator(
        #     content_filter=BM25ContentFilter(
        #         user_query="WHEN_WE_FOCUS_BASED_ON_A_USER_QUERY", 
        #         bm25_threshold=1.0
        #     )
        # ),
    )
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(
            url="https://docs.micronaut.io/4.9.9/guide/",
            config=run_config
        )
        print("原始Markdown长度:", len(result.markdown.raw_markdown))
        print("优化Markdown长度:", len(result.markdown.fit_markdown))

if __name__ == "__main__":
    asyncio.run(main())
```

### JavaScript执行和结构化数据提取
```python
import asyncio
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai import JsonCssExtractionStrategy
import json

async def extract_courses():
    # 定义提取模式
    schema = {
        "name": "KidoCode课程",
        "baseSelector": "section.charge-methodology .w-tab-content > div",
        "fields": [
            {
                "name": "section_title",
                "selector": "h3.heading-50",
                "type": "text",
            },
            {
                "name": "section_description",
                "selector": ".charge-content",
                "type": "text",
            },
            {
                "name": "course_name",
                "selector": ".text-block-93",
                "type": "text",
            },
            {
                "name": "course_description",
                "selector": ".course-content-text",
                "type": "text",
            },
            {
                "name": "course_icon",
                "selector": ".image-92",
                "type": "attribute",
                "attribute": "src"
            }
        ]
    }

    # 创建CSS提取策略
    extraction_strategy = JsonCssExtractionStrategy(schema, verbose=True)

    browser_config = BrowserConfig(
        headless=False,
        verbose=True
    )
    run_config = CrawlerRunConfig(
        extraction_strategy=extraction_strategy,
        js_code=["""(async () => {
            const tabs = document.querySelectorAll("section.charge-methodology .tabs-menu-3 > div");
            for(let tab of tabs) {
                tab.scrollIntoView();
                tab.click();
                await new Promise(r => setTimeout(r, 500));
            }
        })();"""],
        cache_mode=CacheMode.BYPASS
    )
        
    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(
            url="https://www.kidocode.com/degrees/technology",
            config=run_config
        )

        courses = json.loads(result.extracted_content)
        print(f"成功提取 {len(courses)} 个课程")
        print(json.dumps(courses[0], indent=2))

if __name__ == "__main__":
    asyncio.run(extract_courses())
```

### 带持久配置的自定义浏览器
```python
import os, sys
from pathlib import Path
import asyncio, time
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

async def test_news_crawl():
    # 创建持久化用户数据目录
    user_data_dir = os.path.join(Path.home(), ".crawl4ai", "browser_profile")
    os.makedirs(user_data_dir, exist_ok=True)

    browser_config = BrowserConfig(
        verbose=True,
        headless=True,
        user_data_dir=user_data_dir,
        use_persistent_context=True,
    )
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS
    )
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        url = "https://news.ycombinator.com"  # 示例网站
        
        result = await crawler.arun(
            url,
            config=run_config,
            magic=True,
        )
        
        print(f"成功爬取 {url}")
        print(f"内容长度: {len(result.markdown)}")

if __name__ == "__main__":
    asyncio.run(test_news_crawl())
```

## 与LLM集成

### 使用LLM进行结构化数据提取
```python
import os
import asyncio
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, LLMConfig
from crawl4ai import LLMExtractionStrategy
from pydantic import BaseModel, Field

class OpenAIModelFee(BaseModel):
    model_name: str = Field(..., description="OpenAI模型名称")
    input_fee: str = Field(..., description="输入token的费用")
    output_fee: str = Field(..., description="输出token的费用")

async def extract_openai_pricing():
    browser_config = BrowserConfig(verbose=True)
    run_config = CrawlerRunConfig(
        word_count_threshold=1,
        extraction_strategy=LLMExtractionStrategy(
            # 这里可以使用Litellm库支持的任何提供者，例如：ollama/qwen2
            # provider="ollama/qwen2", api_token="no-token", 
            llm_config = LLMConfig(provider="openai/gpt-4o", api_token=os.getenv('OPENAI_API_KEY')), 
            schema=OpenAIModelFee.schema(),
            extraction_type="schema",
            instruction="""从爬取的内容中提取所有提到的模型名称及其输入和输出token的费用。
            不要遗漏整个内容中的任何模型。一个提取的模型JSON格式应如下所示：
            {"model_name": "GPT-4", "input_fee": "US$10.00 / 1M tokens", "output_fee": "US$30.00 / 1M tokens"}。"""
        ),            
        cache_mode=CacheMode.BYPASS,
    )
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(
            url='https://openai.com/api/pricing/',
            config=run_config
        )
        print(result.extracted_content)

if __name__ == "__main__":
    asyncio.run(extract_openai_pricing())
```

### 智能表格提取
```python
from crawl4ai import LLMTableExtraction, LLMConfig

# 配置智能表格提取
table_strategy = LLMTableExtraction(
    llm_config=LLMConfig(provider="openai/gpt-4.1-mini"),
    enable_chunking=True,           # 处理大型表格
    chunk_token_threshold=5000,     # 智能分块阈值
    overlap_threshold=100,          # 块间保持上下文
    extraction_type="structured"    # 获取结构化数据输出
)

config = CrawlerRunConfig(table_extraction_strategy=table_strategy)
result = await crawler.arun("https://complex-tables-site.com", config=config)

# 表格自动分块、处理和合并
for table in result.tables:
    print(f"提取表格: {len(table['data'])} 行")
```

### 搜索引擎结果页面提取
```python
from crawl4ai import LLMExtractionStrategy
from pydantic import BaseModel, Field

class SearchResult(BaseModel):
    title: str = Field(..., description="搜索结果标题")
    url: str = Field(..., description="搜索结果URL")
    snippet: str = Field(..., description="搜索结果摘要")
    rank: int = Field(..., description="搜索结果排名")

async def extract_search_results():
    schema = SearchResult.schema()
    
    extraction_strategy = LLMExtractionStrategy(
        llm_config=LLMConfig(provider="openai/gpt-3.5-turbo"),
        schema=schema,
        extraction_type="schema",
        instruction="从搜索结果页面中提取所有搜索结果，包括标题、URL、摘要和排名。"
    )
    
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url="https://www.google.com/search?q=crawl4ai",
            config=CrawlerRunConfig(
                extraction_strategy=extraction_strategy
            )
        )
        print(result.extracted_content)
```

## API调用

### REST API示例
```python
import requests

# 提交爬取任务
response = requests.post(
    "http://localhost:11235/crawl",
    json={
        "urls": ["https://example.com"], 
        "priority": 10,
        "config": {
            "cache_mode": "enabled",
            "word_count_threshold": 10,
            "extraction_strategy": {
                "type": "css",
                "schema": {
                    "name": "Example Content",
                    "baseSelector": "body",
                    "fields": [
                        {
                            "name": "title",
                            "selector": "h1",
                            "type": "text"
                        }
                    ]
                }
            }
        }
    }
)

if response.status_code == 200:
    print("爬取任务提交成功。")
    
    if "results" in response.json():
        results = response.json()["results"]
        print("爬取任务完成。结果：")
        for result in results:
            print(result)
    else:
        task_id = response.json()["task_id"]
        print(f"爬取任务已提交。任务ID: {task_id}")
        result = requests.get(f"http://localhost:11235/task/{task_id}")
```

### Docker客户端API
```python
from crawl4ai.docker_client import Crawl4aiDockerClient

async def docker_crawl():
    client = Crawl4aiDockerClient(base_url="http://localhost:11235")
    results = await client.crawl(
        urls=["https://httpbin.org/html"],
        hooks={
            "on_page_context_created": on_page_context_created,
            "before_goto": before_goto
        }
    )
    print(results)

# 定义钩子函数
async def on_page_context_created(page, context, **kwargs):
    """阻止图像加载以加速爬取"""
    await context.route("**/*.{png,jpg,jpeg,gif,webp}", lambda route: route.abort())
    await page.set_viewport_size({"width": 1920, "height": 1080})
    return page

async def before_goto(page, context, url, **kwargs):
    """添加自定义头"""
    await page.set_extra_http_headers({'X-Crawl4AI': 'v0.7.5'})
    return page
```

## 多网站爬取实现

### 批量网站爬取
```python
import asyncio
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

class MultiSiteCrawler:
    def __init__(self, max_concurrent=5):
        self.max_concurrent = max_concurrent
        self.browser_config = BrowserConfig(headless=True, verbose=False)
        
    async def crawl_single_site(self, url):
        """爬取单个网站"""
        try:
            async with AsyncWebCrawler(config=self.browser_config) as crawler:
                result = await crawler.arun(
                    url=url,
                    config=CrawlerRunConfig(cache_mode=CacheMode.ENABLED)
                )
                
                if result.success:
                    return {
                        "url": url,
                        "status": "success",
                        "content_length": len(result.markdown),
                        "markdown": result.markdown,
                        "error": None
                    }
                else:
                    return {
                        "url": url,
                        "status": "failed",
                        "content_length": 0,
                        "markdown": "",
                        "error": result.error_message
                    }
        except Exception as e:
            return {
                "url": url,
                "status": "error",
                "content_length": 0,
                "markdown": "",
                "error": str(e)
            }
    
    async def crawl_multiple_sites(self, urls):
        """批量爬取多个网站"""
        # 限制并发数量
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def crawl_with_semaphore(url):
            async with semaphore:
                return await self.crawl_single_site(url)
        
        tasks = [crawl_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return results

# 使用示例
async def main():
    urls = [
        "https://www.github.com",
        "https://www.stackoverflow.com", 
        "https://www.python.org",
        "https://www.djangoproject.com",
        "https://flask.palletsprojects.com"
    ]
    
    crawler = MultiSiteCrawler(max_concurrent=3)
    results = await crawler.crawl_multiple_sites(urls)
    
    for result in results:
        if isinstance(result, dict):
            print(f"URL: {result['url']}, Status: {result['status']}, Length: {result['content_length']}")
        else:
            print(f"Error: {result}")

if __name__ == "__main__":
    asyncio.run(main())
```

### 多网站结构化数据提取
```python
import asyncio
import json
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, LLMConfig
from crawl4ai import LLMExtractionStrategy
from pydantic import BaseModel, Field

class WebsiteInfo(BaseModel):
    site_name: str = Field(..., description="网站名称")
    main_title: str = Field(..., description="网站主标题")
    description: str = Field(..., description="网站描述")
    main_topics: list[str] = Field(..., description="主要话题或内容")

class MultiSiteStructuredExtractor:
    def __init__(self, llm_provider="openai/gpt-3.5-turbo", api_token=None):
        self.llm_config = LLMConfig(provider=llm_provider, api_token=api_token)
        self.browser_config = BrowserConfig(headless=True, verbose=False)
        
    async def extract_from_single_site(self, url):
        """从单个网站提取结构化信息"""
        try:
            extraction_strategy = LLMExtractionStrategy(
                llm_config=self.llm_config,
                schema=WebsiteInfo.schema(),
                extraction_type="schema",
                instruction="""从网站内容中提取网站名称、主标题、描述和主要话题。
                如果无法找到某些信息，使用空字符串。"""
            )
            
            async with AsyncWebCrawler(config=self.browser_config) as crawler:
                result = await crawler.arun(
                    url=url,
                    config=CrawlerRunConfig(
                        extraction_strategy=extraction_strategy,
                        cache_mode=CacheMode.ENABLED
                    )
                )
                
                if result.success and result.extracted_content:
                    extracted_data = json.loads(result.extracted_content)
                    return {
                        "url": url,
                        "status": "success",
                        "extracted_data": extracted_data,
                        "error": None
                    }
                else:
                    return {
                        "url": url,
                        "status": "failed",
                        "extracted_data": None,
                        "error": result.error_message if result else "Unknown error"
                    }
        except Exception as e:
            return {
                "url": url,
                "status": "error",
                "extracted_data": None,
                "error": str(e)
            }
    
    async def extract_from_multiple_sites(self, sites_config):
        """从多个网站提取结构化信息"""
        semaphore = asyncio.Semaphore(3)  # 限制并发
        
        async def extract_with_semaphore(site_config):
            async with semaphore:
                url = site_config["url"]
                return await self.extract_from_single_site(url)
        
        tasks = [extract_with_semaphore(config) for config in sites_config]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return results

# 使用示例
async def main():
    sites_config = [
        {
            "url": "https://www.python.org",
        },
        {
            "url": "https://www.djangoproject.com",
        },
        {
            "url": "https://flask.palletsprojects.com",
        }
    ]
    
    extractor = MultiSiteStructuredExtractor()
    results = await extractor.extract_from_multiple_sites(sites_config)
    
    for result in results:
        if isinstance(result, dict):
            print(f"URL: {result['url']}, Status: {result['status']}")
            if result['extracted_data']:
                print(f"Extracted: {json.dumps(result['extracted_data'], indent=2, ensure_ascii=False)}")
        else:
            print(f"Error: {result}")

if __name__ == "__main__":
    asyncio.run(main())
```

### 带LLM处理的批量爬取
```python
import asyncio
import openai
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

class LLMProcessor:
    def __init__(self, api_key, model="gpt-3.5-turbo"):
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model
    
    async def summarize_content(self, content, max_length=200):
        """使用LLM总结内容"""
        prompt = f"""
        请用中文总结以下内容，保持原文的主要信息，字数控制在{max_length}字以内：
        
        {content[:4000]}  # 限制输入长度
        """
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"总结失败: {str(e)}"
    
    async def classify_content(self, content, categories):
        """使用LLM分类内容"""
        prompt = f"""
        请将以下内容分类到以下类别之一：{', '.join(categories)}
        
        内容：
        {content[:2000]}
        
        请只返回类别名称。
        """
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20
            )
            category = response.choices[0].message.content.strip()
            # 验证分类结果是否在预定义类别中
            if category in categories:
                return category
            else:
                # 如果返回的不是有效类别，返回最接近的匹配
                return categories[0]  # 或者根据需要实现更复杂的匹配逻辑
        except Exception as e:
            return f"分类失败: {str(e)}"

class AdvancedMultiSiteCrawler:
    def __init__(self, llm_api_key, max_concurrent=3):
        self.llm_processor = LLMProcessor(api_key=llm_api_key)
        self.browser_config = BrowserConfig(headless=True, verbose=False)
        self.max_concurrent = max_concurrent
    
    async def process_single_site(self, site_info):
        """处理单个网站：爬取、总结、分类"""
        url = site_info["url"]
        categories = site_info.get("categories", ["科技", "新闻", "教育", "商业"])
        
        try:
            # 1. 爬取网站
            async with AsyncWebCrawler(config=self.browser_config) as crawler:
                result = await crawler.arun(
                    url=url,
                    config=CrawlerRunConfig(cache_mode=CacheMode.ENABLED)
                )
                
            if not result.success:
                return {
                    "url": url,
                    "status": "crawling_failed",
                    "error": result.error_message,
                    "summary": "",
                    "category": "",
                    "content_length": 0
                }
            
            content = result.markdown
            
            # 2. 并行执行LLM任务
            summary_task = self.llm_processor.summarize_content(content)
            category_task = self.llm_processor.classify_content(content, categories)
            
            summary, category = await asyncio.gather(summary_task, category_task)
            
            return {
                "url": url,
                "status": "success",
                "summary": summary,
                "category": category,
                "content_length": len(content),
                "original_content_preview": content[:200] + "..." if len(content) > 200 else content
            }
            
        except Exception as e:
            return {
                "url": url,
                "status": "error",
                "error": str(e),
                "summary": "",
                "category": "",
                "content_length": 0
            }
    
    async def process_multiple_sites(self, sites_info):
        """批量处理多个网站"""
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def process_with_semaphore(site_info):
            async with semaphore:
                return await self.process_single_site(site_info)
        
        tasks = [process_with_semaphore(site_info) for site_info in sites_info]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return results

# 使用示例
async def main():
    sites_info = [
        {
            "url": "https://www.techcrunch.com",
            "categories": ["科技", "创业", "投资"]
        },
        {
            "url": "https://www.bbc.com/news",
            "categories": ["新闻", "国际", "政治"]
        },
        {
            "url": "https://www.coursera.org",
            "categories": ["教育", "在线课程", "学习"]
        }
    ]
    
    # 注意：需要设置您的OpenAI API密钥
    crawler = AdvancedMultiSiteCrawler(llm_api_key="your-openai-api-key-here")
    results = await crawler.process_multiple_sites(sites_info)
    
    for result in results:
        if isinstance(result, dict):
            print(f"\n网站: {result['url']}")
            print(f"状态: {result['status']}")
            if result['status'] == 'success':
                print(f"分类: {result['category']}")
                print(f"总结: {result['summary']}")
                print(f"内容长度: {result['content_length']} 字符")
            else:
                print(f"错误: {result.get('error', 'Unknown error')}")
        else:
            print(f"处理错误: {result}")

if __name__ == "__main__":
    # 确保设置了OpenAI API密钥
    # os.environ["OPENAI_API_KEY"] = "your-api-key-here"
    asyncio.run(main())
```

## 最佳实践

### 1. 性能优化
- 使用缓存模式避免重复爬取
- 设置适当的并发数量
- 合理使用浏览器配置（headless模式、资源限制）
- 利用CDN或代理服务

### 2. 错误处理
```python
import asyncio
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

async def robust_crawl(url, max_retries=3):
    """带重试机制的健壮爬取"""
    last_error = None
    
    for attempt in range(max_retries):
        try:
            async with AsyncWebCrawler(config=BrowserConfig(headless=True)) as crawler:
                result = await crawler.arun(
                    url=url,
                    config=CrawlerRunConfig(
                        cache_mode=CacheMode.ENABLED,
                        timeout=30000  # 30秒超时
                    )
                )
                
                if result.success:
                    return result
                else:
                    last_error = result.error_message
                    
        except Exception as e:
            last_error = str(e)
            
        print(f"尝试 {attempt + 1} 失败: {last_error}")
        
        if attempt < max_retries - 1:
            await asyncio.sleep(2 ** attempt)  # 指数退避
    
    raise Exception(f"爬取失败，已尝试 {max_retries} 次: {last_error}")
```

### 3. 资源管理
```python
import asyncio
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

class ManagedCrawler:
    def __init__(self):
        self.crawler = None
    
    async def __aenter__(self):
        self.crawler = AsyncWebCrawler(config=BrowserConfig(headless=True))
        await self.crawler.__aenter__()
        return self.crawler
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.crawler:
            await self.crawler.__aexit__(exc_type, exc_val, exc_tb)
```

### 4. 遵守robots.txt和使用率限制
- 在生产环境中添加适当的延迟
- 遵守网站的robots.txt文件
- 使用代理轮换避免IP被封
- 实现智能重试机制处理反爬虫措施

### 5. 数据验证和清洗
```python
def validate_and_clean_data(data):
    """验证和清洗提取的数据"""
    if not data:
        return None
    
    # 去除多余空白字符
    if isinstance(data, str):
        data = data.strip()
        
    # 验证URL格式（如果是URL字段）
    # 验证日期格式等
    
    return data
```

## 指导来源和依据

本文档中的信息来源于以下资源：

1. **主要来源**: Crawl4AI GitHub仓库
   - 来源URL: https://github.com/unclecode/crawl4ai
   - 使用方法的依据: 从GitHub仓库中提取的安装说明、API文档、示例代码和功能描述

2. **技术文档**:
   - 从官方GitHub仓库中获取的API参考
   - 提取策略和配置选项的说明
   - 浏览器配置和爬取选项的详细说明

3. **代码示例**:
   - 从GitHub仓库中的示例代码提取并改编
   - 包括基本爬取、结构化数据提取、LLM集成等示例

通过以上文档，您可以使用Crawl4AI对多个网站进行爬取，并利用LLM对爬取的数据进行处理。文档涵盖了从基础安装到高级特性的全方位内容，包括实际的代码示例，让您能够快速上手并构建强大的网络爬取和数据处理系统。