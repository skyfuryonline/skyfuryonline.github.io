---
layout: post
title: "爬虫性能优化：从串行到并发，让AI摘要速度起飞"
subtitle: "通过asyncio.gather，将LLM API调用效率提升N倍的实战记录"
date: 2025-11-04 18:00:00
author: "LH"
tags: [Python, asyncio, Performance, LLM, API]
group: blog-development
---

## 前言：当系统“能用”之后，我们追求“好用”

在此前的系列教程中，我们已经构建了一个功能完备的自动化信息聚合系统。它能自动抓取、自动总结、自动部署。但是，当爬取的新文章数量增多时，一个新的瓶颈出现了：**LLM 总结的速度**。

我们最初的实现方式是“串行”的：总结完一篇文章，再开始总结下一篇。如果每篇文章的总结需要 10 秒，那么 5 篇文章就需要 50 秒。这极大地延长了我们 CI/CD 工作流的运行时间。

这篇收官之作，我们将聚焦于性能优化，一步步地、可复现地展示如何将 LLM API 的调用效率提升 N 倍，让我们的系统真正“起飞”。

## 瓶颈分析：串行 vs. 并发

在处理多个独立的网络请求时，我们通常有两种模式：

1.  **串行 (Serial)**: 一次只做一个任务，做完再做下一个。这是我们优化前的实现，简单、健壮，但效率最低。
    *   *总耗时 = 任务A耗时 + 任务B耗时 + ...*

2.  **并发 (Concurrent)**: 同时发起多个独立的请求，然后等待它们全部完成。这就像雇佣了多个工人同时干活，是 I/O 密集型任务（如网络请求）的最佳优化方案。
    *   *总耗时 ≈ 耗时最长的那个任务的时间*

我们的目标，就是将串行处理，改造为**并发处理**。

## 第一步：改造“工人” - 将 `summarizer` 异步化

要实现并发，首先需要我们的“工人”（`get_summary` 函数）支持异步操作。这意味着它在等待网络响应时，不能阻塞整个程序，而是应该把控制权交还给事件循环，让 CPU 去处理其他任务。

**文件定位**: `llm/summarizer.py`

**核心修改**: 将 `openai` 的同步客户端，替换为异步客户端 `AsyncOpenAI`，并改造 `get_summary` 为 `async def`。

**修改前 (伪代码):**
```python
# llm/summarizer.py (Before)
from openai import OpenAI

client = OpenAI(...)

def get_summary(...):
    response = client.chat.completions.create(...)
    return response.choices[0].message.content
```

**修改后 (完整代码):**
```python
# llm/summarizer.py (After)

import os
import asyncio
from openai import AsyncOpenAI # 1. 导入异步客户端

client = None

def initialize_client():
    """Initializes the AsyncOpenAI client, reusing it for efficiency."""
    global client
    api_key = os.environ.get('LLM_API_KEY')
    base_url = os.environ.get('LLM_API_BASE_URL')
    if api_key and client is None:
        # 2. 实例化异步客户端
        client = AsyncOpenAI(api_key=api_key, base_url=base_url)

# 3. 将函数声明为 async def
async def get_summary(content: str, model: str, prompt_template: str) -> str:
    """Asynchronously calls an OpenAI-compatible LLM to get a summary."""
    initialize_client()
    
    if not client:
        return "Error: LLM client not initialized. Check LLM_API_KEY."
    if not content:
        return "(Content was empty, no summary generated)"

    try:
        # 4. 使用 await 调用异步方法
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt_template},
                {"role": "user", "content": content[:15000]}
            ],
            temperature=0.5,
            timeout=180
        )
        summary = response.choices[0].message.content.strip()
        return summary
    except Exception as e:
        return f"Error calling LLM API: {e}"
```
通过这四步，我们的 `get_summary` 函数就从一个“一次只能干一件事”的同步工人，升级为了一个“能随时响应新任务”的异步工人。

## 第二步：改造“包工头” - `main.py` 实现并发调度

现在，我们需要改造“包工头”（`main.py`），让它能同时给多个“工人”派发任务，而不是一个一个地等。这需要用到 `asyncio.gather`，它是 Python 中实现并发的“神器”。

**文件定位**: `crawlers/main.py`

**核心修改**: 在 LLM 集成部分，不再直接 `await` `get_summary`，而是先将其收集到任务列表中，最后使用 `asyncio.gather` 一次性执行。

**修改前 (伪代码):**
```python
# crawlers/main.py (Before)

# ...
for article in articles_metadata:
    if article['link'] not in summarized_urls:
        # ... 读取 content ...
        # 串行调用，一次只能等一个
        summary = get_summary(content, ...)
        article['summary'] = summary
# ...
```

**修改后 (完整代码):**
```python
# crawlers/main.py (After, in the main() function)

# ... (省略了爬虫部分的循环)

# --- LLM Integration (Concurrent) --- #
llm_profile_name = site.get("llm_profile")
if llm_profile_name and llm_profile_name in llm_profiles:
    profile = llm_profiles[llm_profile_name]
    print(f"Summarizing new articles using LLM profile: '{llm_profile_name}'")

    tasks = []
    articles_to_summarize = []

    # 1. 第一轮循环：收集所有需要执行的任务
    for article in articles_metadata:
        if article['link'] not in summarized_urls:
            try:
                with open(os.path.join(article['cache_path'], 'content.txt'), 'r', encoding='utf-8') as content_file:
                    content = content_file.read()
                
                # 创建一个任务协程，但不立即执行
                task = get_summary(content, profile['model'], profile['prompt'])
                tasks.append(task)
                articles_to_summarize.append(article)
            except Exception as e:
                article['summary'] = f"Failed to read content for summary: {e}"

    # 2. 使用 asyncio.gather 并发执行所有任务
    if tasks:
        print(f"Running {len(tasks)} summarization tasks concurrently...")
        summaries = await asyncio.gather(*tasks)
        print("Summarization tasks finished.")

        # 3. 第二轮循环：将返回的结果一一对应地赋值回去
        for i, summary_result in enumerate(summaries):
            articles_to_summarize[i]['summary'] = summary_result
            print(f"  - Summarized: {articles_to_summarize[i]['title']}")

all_articles_metadata.extend(articles_metadata)
# ...
```
这个新的逻辑非常清晰：
1.  **分发任务**: 遍历所有新文章，创建一堆摘要任务（`tasks`），并记下这些任务对应的文章（`articles_to_summarize`）。
2.  **同时开工**: 使用 `await asyncio.gather(*tasks)`，告诉系统：“把列表里所有的活儿，能同时干的都给我干起来！”
3.  **收集成果**: `gather` 会等到所有任务都完成后，将所有摘要结果，按我们当初添加任务的顺序，一次性地返回一个列表 `summaries`。
4.  **论功行赏**: 最后，我们再遍历这个 `summaries` 列表，把它里面的结果，按顺序放回对应的文章元数据中。

## 最终成果：从项目到产品

通过这次并发优化，我们的自动化系统，在功能、健壮性、UI/UX 和运行效率上，都达到了一个非常成熟的水平。它不再仅仅是一个“玩具项目”，而是一个真正高效、可用的“产品”。

这个从 0 到 1，再从 1 到 100 的过程，充满了各种预想不到的挑战。但每一次挑战，都让我们对软件工程的理解更深一层。希望这个系列教程，能为你未来的项目提供宝贵的经验和启示。
