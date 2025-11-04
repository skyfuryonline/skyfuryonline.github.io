# Crawl4AI 使用指南（v0.7.6 修订版）

> 基于 Crawl4AI v0.7.6（稳定版）编写。
> 发布重点：0.7.6 引入对 Docker Job Queue API 的完整 **Webhook** 支持，改进若干兼容性与稳定性问题。

---

## 目录
1. [简介](#简介)
2. [重要更新（v0.7.6）](#重要更新v076)
3. [安装](#安装)
4. [基本用法](#基本用法)
5. [高级特性](#高级特性)
6. [网站爬取技术](#网站爬取技术)
7. [与LLM集成](#与llm集成)
8. [API调用（含 Docker Job API 与 Webhook）](#ap调用含-docker-job-api-与-webhook)
9. [多网站爬取实现](#多网站爬取实现)
10. [最佳实践](#最佳实践)
11. [常见问题 FAQ](#常见问题-faq)
12. [升级与迁移指南](#升级与迁移指南)
13. [指导来源与依据](#指导来源与依据)

---

## 简介

Crawl4AI 是一款为 LLM 场景优化的开源异步爬虫框架，擅长将网页内容整理为结构化或 Markdown 格式，方便后续用于 RAG、语义检索或数据管道。

## 重要更新（v0.7.6）

v0.7.6 的主要改动和新增特性摘要（应在升级前阅读）：

- **Docker Job Queue API 的 Webhook 完整支持**：现在 `/crawl/job` 与 `/llm/job` 等队列任务可以配置 Webhook，实现实时回调通知；支持自定义 HTTP 头、包含完整任务数据的负载，以及带指数退避的自动重试策略，减少轮询需求。
- **Docker 镜像与部署**：官方发布 `unclecode/crawl4ai:0.7.6` 镜像，`latest` 标签已指向稳定的 0.7.6 版本。
- **若干兼容性修复与弃用说明**：修复了会话泄漏、重复访问、URL 规范化等问题；保留若干向下兼容的弃用别名（如老的 Markdown 生成器名会 alias 到 `DefaultMarkdownGenerator` 并给出警告）。

> 请在部署生产环境前务必阅读“升级与迁移指南”章节以避免运行时错误。

## 安装

### 基础安装（推荐异步版本）
```bash
pip install -U crawl4ai
# 初始化 Playwright 浏览器（若自动安装失败）
python -m playwright install --with-deps chromium
```

**说明**：库默认推荐使用异步接口 `AsyncWebCrawler`。同步支持（`crawl4ai[sync]`）已标记为弃用并计划在未来版本移除；若非特殊需求，请迁移到异步实现。

### 安装可选功能
```bash
pip install crawl4ai[torch]
pip install crawl4ai[transformer]
pip install crawl4ai[cosine]
# 全部可选功能
pip install crawl4ai[all]
```

### 开发版安装
```bash
git clone https://github.com/unclecode/crawl4ai.git
cd crawl4ai
pip install -e .
# 可按需添加可选功能
pip install -e '.[torch]'
```

### Docker 部署
```bash
# 推荐拉取指定版本
docker pull unclecode/crawl4ai:0.7.6
# 或使用 latest（指向 v0.7.6 稳定版）
docker pull unclecode/crawl4ai:latest
# 运行示例
docker run -d -p 11235:11235 --name crawl4ai --shm-size=1g unclecode/crawl4ai:0.7.6
```

## 基本用法

（异步 API 示例，推荐）

```python
import asyncio
from crawl4ai import AsyncWebCrawler

async def main():
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url="https://www.nbcnews.com/business")
        print(result.markdown)

if __name__ == "__main__":
    asyncio.run(main())
```

## 高级特性

- Markdown 生成、结构化提取、浏览器会话管理、媒体抓取、无限滚动处理、截图与原始 HTML 保存等。
- 注意：老的 Markdown 生成器名仍可用但会触发弃用警告；推荐统一使用 `DefaultMarkdownGenerator`。

## 网站爬取技术

本节示例展示如何配置 `BrowserConfig`、`CrawlerRunConfig`、内容过滤器和提取策略（BM25、启发式裁剪等）。示例代码与 v0.7.x 系列兼容。

## 与LLM集成

- 支持基于 LLM 的结构化提取（`LLMExtractionStrategy`）、表格抽取、搜索结果抽取等。请在生产环境中注意 LLM 调用的并发和费用控制。
- LLM 配置请使用 `LLMConfig` 管理 provider 与凭证，优先使用环境变量或机密管理服务传递 API 密钥。

## API调用（含 Docker Job API 与 Webhook）

### 提交爬取任务（REST）
```python
import requests
response = requests.post("http://localhost:11235/crawl", json={
    "urls": ["https://example.com"],
    "priority": 10,
})
```

### Docker Job Queue + Webhook（v0.7.6）
- v0.7.6 为 Docker Job Queue API 引入了 **Webhook**：可在 `config.yml` 或任务级配置 webhook 回调 URL。回调支持自定义 HTTP 头部，并可以在负载中包含完整任务结果，避免轮询 `GET /task/{id}`。
- Webhook 交付具备**自动重试与指数退避**机制，减轻接收端瞬时压力。

> 使用场景示例：大规模批量爬取任务完成后，Docker 容器可通过 Webhook 将结果推送至下游服务（如消息队列、数据处理管道或通知系统）。

## 多网站爬取实现

- 使用 `AsyncWebCrawler` 与 `asyncio.Semaphore` 控制并发。
- 建议限制并发 Playwright 实例数量并合理设置缓存与会话持久化（`user_data_dir`）以减轻资源消耗。

## 最佳实践

- 使用缓存模式避免重复抓取（并在容器中显式设置缓存目录以持久化）。
- 合理设置并发与超时，使用钩子函数（`before_goto`, `on_page_context_created`）优化请求。
- 严格遵守 robots.txt 与访问频率限制，避免 IP 被封。

## 常见问题 FAQ

- **Playwright 安装失败**：在某些平台上可改用 `python -m playwright install chromium`。
- **为什么拿不到动态渲染内容？**：检查 `js_code`、视口、用户代理与是否触发了反爬机制（需考虑 stealth/undetected 浏览器方案）。
- **同步 API 是否可用？**：同步版本为兼容选项，但已弃用，建议迁移到异步接口。

## 升级与迁移指南

升级到 v0.7.6 前建议注意：

1. **备份配置与缓存目录**。
2. **检查自定义导入路径**：旧的 `crawl4ai/browser/*` 模块在若些版本中被重构，若直接 `from crawl4ai.browser...`，请更新为新 pooled browser 模块或检查 release notes。 
3. **如果覆盖了内部策略接口**（例如重写 `AsyncPlaywrightCrawlerStrategy.get_page`），请对照 v0.7.6 的函数签名进行适配。
4. **Docker 镜像**：拉取 `unclecode/crawl4ai:0.7.6` 并在测试环境运行完整回归。若使用 `latest` 标签，注意它已指向 0.7.6（截至本指南编写时）。

## 指导来源与依据

- 官方 GitHub Releases 与源码仓库（unclecode/crawl4ai）
- 官方文档（Docker 部署与博客）
- 官方 Docker Hub 镜像清单


---

*文档说明：本修订为基于 v0.7.6 的结构化更新，主要补充了 Webhook 与 Docker Job Queue 的使用说明，并将文中与旧版不兼容或已弃用的部分标注为注意项。若需要我可以把完整的旧版到新版逐条差异（CHANGELOG）整理为补丁格式供你核对。*
