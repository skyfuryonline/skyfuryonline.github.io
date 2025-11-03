# 爬虫与博客集成计划

**核心目标：**
打造一个全自动的信息聚合与发布系统。该系统通过 GitHub Actions 定时运行一个爬虫，抓取指定网站的信息，调用 LLM API 对信息进行总结，最后将结果自动更新并发布到博客的 "Daily" 页面。

---

### 第一阶段：环境准备与基础爬虫

这是构建整个系统的基础。

1.  **创建新分支**: 创建一个 `feature/crawler-integration` 分支，用于开发此功能。
2.  **修改 GitHub Actions 工作流 (`deploy.yml`)**:
    *   **加入 Python 环境**: 在 `build` 任务中，增加 `actions/setup-python@v4` 步骤，为运行爬虫准备好 Python 环境。
    *   **安装 Python 依赖**: 增加一步，通过 `pip install -r requirements.txt` 来安装爬虫所需的库（如 `requests`, `beautifulsoup4` 等）。
    *   **创建 `requirements.txt`**: 在项目根目录创建 `requirements.txt` 文件，并写入爬虫需要的 Python 库。
3.  **创建爬虫基础框架**:
    *   在项目根目录下创建一个 `scripts` 文件夹。
    *   在 `scripts` 文件夹中，创建一个 `crawler.py` 文件。
    *   **定义数据结构**: 在 `scripts` 文件夹中，创建一个 `config.json` 文件。这个文件将按照您的设想，定义要爬取的 `url` 列表和对应的 `parsing_rules` (解析规则，例如 CSS 选择器)。
    *   **编写基础爬虫逻辑**: 在 `crawler.py` 中，编写基础代码，使其能够：
        *   读取 `config.json`。
        *   遍历 URL 列表，下载网页内容。
        *   根据解析规则，提取所需信息。
        *   将提取到的原始信息，以一种结构化的格式（例如 JSON），保存到 `_data/daily_info.json` 文件中。

4.  **修改 GitHub Actions 工作流以运行爬虫**:
    *   在 `jekyll build` 之前，增加一步 `run: python scripts/crawler.py` 来执行爬虫脚本。
    *   **实现“每日一次”逻辑**: 为了满足您“推送时不总运行爬虫”的需求，我们将利用 GitHub Actions 的 `if` 条件判断。
        *   只有当工作流是由于 `schedule` (定时) 或 `workflow_dispatch` (手动) 触发时，才运行爬虫步骤。
        *   当工作流是由于 `push` 触发时，跳过爬虫步骤。

**此阶段完成后，您的 Actions 将具备能力：每天自动抓取原始数据并存入 `_data` 文件夹。**

---

### 第二阶段：集成 LLM API 与数据展示

1.  **集成 LLM API**:
    *   **添加 API 密钥**: 您需要将您的 LLM API 密钥，以 **Secret** 的形式添加到您 GitHub 仓库的 `Settings > Secrets and variables > Actions` 中。例如，命名为 `LLM_API_KEY`。
    *   **修改爬虫脚本 (`crawler.py`)**:
        *   在脚本中，增加调用 LLM API 的函数。
        *   在获取到原始信息后，将信息发送给 LLM API，并附上您的“总结”提示 (Prompt)。
        *   接收 LLM 返回的总结内容。
        *   将**原始信息**和**总结内容**一并写入 `_data/daily_info.json`。
    *   **修改 GitHub Actions 工作流**: 在运行爬虫脚本的步骤中，通过 `env` 将您设置的 `LLM_API_KEY` 安全地传递给 Python 脚本。

2.  **在 "Daily" 页面展示数据**:
    *   **修改 `daily.html`**:
        *   使用 `Liquid` 循环，读取 `_data/daily_info.json` 文件中的数据。
        *   设计一个合适的 HTML 结构（例如，卡片式布局），将每个网站的“总结”和“原始信息要点”清晰地展示出来。

**此阶段完成后，您的 "Daily" 页面将能够展示由 LLM 自动总结的、每日更新的信息。**

---

### 第三阶段：优化与完善 (可选)

1.  **错误处理与日志记录**: 增强 `crawler.py`，为网络请求失败、解析错误、API 调用失败等情况添加异常处理和日志记录。
2.  **增量更新**: 优化爬虫逻辑，使其能够判断哪些信息是新的，避免重复抓取和处理。
3.  **前端交互**: 为 "Daily" 页面增加筛选、排序或搜索功能。
