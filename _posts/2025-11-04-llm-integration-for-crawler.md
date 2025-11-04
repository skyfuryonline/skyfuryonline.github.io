---
layout: post
title: "博客爬虫集成(续)：赋予AI摘要能力"
subtitle: "通过LLM API与前端模态框，实现文章的自动总结与优雅展示"
date: 2025-11-04 10:00:00
author: "LH"
tags: [LLM, API, JavaScript, Jekyll, CI/CD]
group: life
---

## 前言：从“是什么”到“讲什么”

在上一篇教程中，我们成功地为博客集成了一个全自动的爬虫，它能每天抓取最新的文章列表。但这只解决了“有什么新内容”的问题。用户看到标题后，依然需要跳转到原文才能判断是否值得一读，这在信息过载的时代效率不高。

我们能否更进一步，让 AI 告诉我们每篇文章**“讲了什么”**？

这篇教程，我们将在此前的爬虫基础上，集成大语言模型（LLM）的强大能力，实现对抓取文章的**自动摘要**功能。最终效果是，用户在 "Daily" 页面点击一篇文章，会先弹出一个由 AI 生成的摘要，帮助用户快速决策，极大地提升信息获取效率。

## 设计哲学：“预生成”与“纯静态”

在纯静态的 GitHub Pages 环境下，我们无法在用户点击时，去实时调用后端服务和 LLM API。因此，我们必须转变思路，采用**“预生成 (Pre-generation)”**的策略。

**核心思想：** 将所有计算密集型、需要后端参与的工作（调用 LLM API），全部前置到**构建阶段 (Build Time)** 完成。

**具体流程：**
1.  爬虫在 Actions 环境中抓取到原文内容。
2.  **立即**将原文内容发送给 LLM API，获取摘要。
3.  将**摘要**和文章标题、链接等元数据**一同**写入最终的 `_data/daily_...json` 文件。
4.  前端页面加载时，摘要信息已经存在于数据中，只需通过 JavaScript 将其展示出来即可。

这个方案完美地契合了静态网站的哲学，保证了用户访问时极致的速度和体验，同时确保了 API 密钥的绝对安全。

## 第一步：API 密钥的安全管理

在与任何需要付费的 API 交互时，安全永远是第一位的。我们绝不能将 API 密钥硬编码在代码中。

**唯一正确的方式：** 使用 GitHub Secrets。

1.  前往您的 GitHub 仓库页面，点击 `Settings` > `Secrets and variables` > `Actions`。
2.  点击 `New repository secret`。
3.  创建一个名为 `LLM_API_KEY` 的 Secret，并将您的 API 密钥粘贴进去。
4.  (可选) 如果您使用代理或自托管服务，可以再创建一个 `LLM_API_BASE_URL` 的 Secret。

这样，密钥就被加密存储了，只有在 Actions 运行时才能被我们的脚本读取。

## 第二步：可扩展的配置 (`config.json`)

为了方便未来为不同网站（如技术博客、论文网站）应用不同的总结模型或指令 (Prompt)，我们将 `config.json` 升级为可配置的“总结策略”。

```json
{
  "sites": [
    {
      "url": "https://www.cnblogs.com/pick/",
      "parser": "CnblogsCrawler",
      "llm_profile": "default_summary" 
    }
  ],
  "llm_profiles": {
    "default_summary": {
      "model": "gpt-3.5-turbo",
      "prompt": "你是一个优秀的内容摘要助手。请将以下文章内容总结为一段150字以内的中文摘要，提取核心观点和主要信息，只输出摘要本身，不要添加任何额外的话。"
    },
    "academic_summary": {
      "model": "gpt-4",
      "prompt": "请将以下论文的核心贡献、使用的方法和最终的实验结果，总结为三点，每点不超过50字。"
    }
  }
}
```
通过 `llm_profiles`，我们可以为不同类型的文章定义不同的总结“配方”，未来只需修改这个 JSON 文件，就能轻松调整总结策略，无需改动任何 Python 代码。

## 第三步：独立的 LLM 调用模块 (`llm/summarizer.py`)

我们创建一个独立的模块，专门负责与 LLM API 通信。这里我们以任何支持 OpenAI 兼容 API 的服务为例。

```python
# llm/summarizer.py

import os
from openai import OpenAI

client = None

def initialize_client():
    """初始化 OpenAI 客户端，以便复用。"""
    global client
    api_key = os.environ.get('LLM_API_KEY')
    base_url = os.environ.get('LLM_API_BASE_URL')
    if api_key and client is None:
        client = OpenAI(api_key=api_key, base_url=base_url)

def get_summary(content: str, model: str, prompt_template: str) -> str:
    """调用 LLM API 获取摘要。"""
    initialize_client()
    
    if not client:
        return "Error: LLM 客户端未初始化，请检查 LLM_API_KEY。"
    if not content:
        return "(文章内容为空，未生成摘要)"

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt_template},
                {"role": "user", "content": content[:15000]} # 截断内容以防超长
            ],
            temperature=0.5,
            timeout=180
        )
        summary = response.choices[0].message.content.strip()
        return summary
    except Exception as e:
        return f"调用 LLM API 时出错: {e}"
```

## 第四步：串联流程 (`crawlers/main.py`)

现在，我们改造“大脑”——`main.py`，让它在爬取到文章后，调用 `summarizer` 来生成摘要。

```python
# crawlers/main.py (核心修改部分)

# ... (在文件顶部导入)
from llm.summarizer import get_summary

async def main():
    # ... (省略了路径设置、加载历史等代码)

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    llm_profiles = config.get("llm_profiles", {})

    all_articles_metadata = []
    for site in config["sites"]:
        # ... (省略了爬虫实例化代码)
        articles_metadata = await crawler_instance.crawl()

        # --- LLM 集成开始 --- #
        llm_profile_name = site.get("llm_profile")
        if llm_profile_name and llm_profile_name in llm_profiles:
            profile = llm_profiles[llm_profile_name]
            print(f"使用 LLM 配置 '{llm_profile_name}' 生成摘要...")
            for article in articles_metadata:
                try:
                    # 1. 读取缓存的原文
                    with open(os.path.join(article['cache_path'], 'content.txt'), 'r', encoding='utf-8') as content_file:
                        content = content_file.read()
                    
                    # 2. 调用 summarizer 获取摘要
                    summary = get_summary(content, profile['model'], profile['prompt'])
                    article['summary'] = summary # 3. 将摘要添加到元数据中
                    print(f"  - 已总结: {article['title']}")
                except Exception as e:
                    article['summary'] = f"生成摘要失败: {e}"
        
        all_articles_metadata.extend(articles_metadata)

    # ... (保存最终数据和清理旧数据的代码)
```

## 第五步：注入 API 密钥 (`deploy.yml`)

最后，我们修改 `.github/workflows/deploy.yml`，通过 `env` 关键字，将我们设置的 Secret 安全地注入到爬虫脚本的运行环境中。

```yaml
      - name: Run crawler 🕷️
        if: github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'
        env:
          LLM_API_KEY: ${{ secrets.LLM_API_KEY }}
          LLM_API_BASE_URL: ${{ secrets.LLM_API_BASE_URL }} # 可选
        run: python crawlers/main.py
```

至此，我们的后端和自动化流水线已经准备就绪。下一次当定时任务运行时，它生成的 `_data/daily_...json` 文件中，将包含一个全新的 `summary` 字段。我们的下一步，就是在前端页面上，将这个摘要优雅地展示给用户。
