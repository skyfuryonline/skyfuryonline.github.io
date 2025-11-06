# DeepAgents 0.2.5 使用说明书

## 概述

DeepAgents 是一个基于 LangGraph 构建的通用"深度代理"库，具有子代理生成、待办事项列表功能和模拟文件系统。它实现了 Claude Code 等高级 AI 应用的关键架构特性，使开发者能够轻松创建用于复杂任务的深度代理。

## 安装

首先，您需要安装 deepagents 库：

```bash
# 使用 pip
pip install deepagents

# 使用 uv (更快的包管理器)
uv add deepagents

# 使用 poetry
poetry add deepagents
```

**注意**: Python 版本要求为 Python >= 3.11, < 4.0

## 核心特性

### 1. 规划与任务分解
- 内置 `write_todos` 工具，使代理能够将复杂任务分解为离散步骤
- 跟踪进度并根据新信息调整计划

### 2. 上下文管理
- 提供文件系统工具（ls、read_file、write_file、edit_file、glob、grep）
- 将大上下文卸载到内存，防止上下文窗口溢出

### 3. 子代理生成
- 内置 task 工具允许代理生成专门的子代理进行上下文隔离
- 保持主代理上下文清洁，同时深入处理特定子任务

### 4. 长期记忆
- 使用 LangGraph 的 Store 扩展代理的跨线程持久内存
- 代理可以从先前对话中保存和检索信息

## 基础使用

### 简单的研究代理示例

```python
# 需要先安装 pip install tavily-python
import os
from typing import Literal
from tavily import TavilyClient
from deepagents import create_deep_agent

# 初始化 Tavily 客户端用于网络搜索
tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

# 创建网络搜索工具
def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """运行网络搜索"""
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )

# 系统提示词，引导代理成为专家研究员
research_instructions = """You are an expert researcher. Your job is to conduct thorough research, and then write a polished report.

You have access to an internet search tool as your primary means of gathering information.

## `internet_search`

Use this to run an internet search for a given query. You can specify the max number of results to return, the topic, and whether raw content should be included.
"""

# 创建深度代理
agent = create_deep_agent(
    tools=[internet_search],
    system_prompt=research_instructions,
)

# 调用代理
result = agent.invoke({"messages": [{"role": "user", "content": "什么是 langgraph？"}]})
print(result)
```

### 使用自定义模型

```python
from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent

# 使用自定义模型
model = init_chat_model("openai:gpt-4o")
agent = create_deep_agent(
    model=model,
)

# 调用代理
result = agent.invoke({"messages": [{"role": "user", "content": "帮我写一个简单的 Python 函数"}]})
print(result)
```

### 使用子代理

```python
import os
from typing import Literal
from tavily import TavilyClient
from deepagents import create_deep_agent

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

# 网络搜索工具
def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """运行网络搜索"""
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )

# 定义子代理
research_subagent = {
    "name": "research-agent",  # 子代理名称
    "description": "用于进行深入的研究问题",  # 子代理描述
    "system_prompt": "你是一个优秀的研究员",  # 子代理系统提示
    "tools": [internet_search],  # 子代理可用的工具
    "model": "openai:gpt-4o",  # 可选：覆盖默认模型
}
subagents = [research_subagent]

# 创建带有子代理的主代理
agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-20250514",
    subagents=subagents
)

# 调用代理
result = agent.invoke({"messages": [{"role": "user", "content": "深入研究 LangGraph 的架构"}]})
print(result)
```

### 中间件配置

```python
from langchain_core.tools import tool
from deepagents import create_deep_agent
from langchain.agents.middleware import AgentMiddleware

# 定义工具
@tool
def get_weather(city: str) -> str:
    """获取某个城市的天气。"""
    return f"城市 {city} 的天气是晴朗的。"

@tool
def get_temperature(city: str) -> str:
    """获取某个城市的温度。"""
    return f"城市 {city} 的气温是 70 华氏度。"

# 自定义中间件
class WeatherMiddleware(AgentMiddleware):
    tools = [get_weather, get_temperature]

# 创建带有中间件的代理
agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-20250514",
    middleware=[WeatherMiddleware()]
)

# 调用代理
result = agent.invoke({"messages": [{"role": "user", "content": "纽约的天气如何？"}]})
print(result)
```

## API 文档

### 主要函数

#### create_deep_agent()
这是创建深度代理的主要函数，支持同步和异步调用。

**参数:**
1. **model**: 默认使用 "claude-sonnet-4-5-20250929"，可传入任何 LangChain 模型对象
2. **system_prompt**: 内置详细系统提示，包含使用内置规划工具、文件系统工具和子代理的说明
3. **tools**: 与工具调用代理一样，可提供代理可访问的工具集
4. **middleware**: 可提供额外的中间件来扩展功能、添加工具或实现自定义钩子
5. **subagents**: 指定代理可交接工作的自定义子代理
6. **interrupt_on**: 配置人类干预点

### 中间件架构

DeepAgents 使用模块化的中间件架构:
1. **TodoListMiddleware**: 提供待办事项工具，帮助代理跟踪任务
2. **FilesystemMiddleware**: 提供文件系统交互工具
3. **SubAgentMiddleware**: 允许提供子代理通过任务工具

## 高级功能

### 人类干预 (HITL - Human-in-the-Loop)

您可以配置人类干预点，让人类用户在代理执行特定工具时进行批准、编辑或拒绝。

```python
from langchain_core.tools import tool
from deepagents import create_deep_agent

@tool
def get_weather(city: str) -> str:
    """获取某个城市的天气。"""
    return f"城市 {city} 的天气是晴朗的。"

# 创建带有中断配置的代理
agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-20250514",
    tools=[get_weather],
    interrupt_on={
        "get_weather": {  # 工具名称
            "allowed_decisions": ["approve", "edit", "reject"]  # 允许的决策
        },
    }
)

# 调用代理（这将在执行 get_weather 工具时中断）
result = agent.invoke({"messages": [{"role": "user", "content": "纽约的天气如何？"}]})
```

### 流式传输

与 LangGraph 代理一样，DeepAgents 支持流式传输输出：

```python
for chunk in agent.stream({"messages": [{"role": "user", "content": "写一个故事"}]}):
    print(chunk)
```

## 文件系统工具

DeepAgents 提供了完整的文件系统操作工具：

- `list_directory` - 列出目录内容
- `read_file` - 读取文件内容
- `write_file` - 写入文件内容
- `edit` - 编辑文件内容
- `grep_search` - 在文件中搜索内容
- `glob` - 使用通配符模式查找文件

## 架构细节

- **基础**: 基于 LangGraph 构建
- **中间件架构**: 模块化设计，可独立使用各个功能
- **文件系统**: 提供短期和长期内存管理
- **子代理系统**: 实现上下文隔离和任务分解
- **规划系统**: 内置待办事项管理

## 许可证

DeepAgents 采用 MIT 许可证。

## 常见问题

### 1. 如何选择合适的模型?
DeepAgents 支持任何 LangChain 兼容的模型。对于复杂任务，推荐使用 claude-sonnet 或 gpt-4 等高级模型。

### 2. 如何处理大型上下文?
使用内置的文件系统工具将大文件存储到内存中，代理可以随时读取和写入这些文件以管理大型上下文。

### 3. 如何调试代理行为?
使用 LangGraph Studio 或流式传输功能来观察代理的思考过程和工具调用。

## 重要说明

- 代理通过 `create_deep_agent` 创建的是 LangGraph 图，可以像使用任何 LangGraph 代理一样与之交互（流式传输、人机交互、内存、studio）
- 项目受到 Claude Code 启发，旨在使其更通用
- 支持 MCP (Model Context Protocol) 工具通过 Langchain MCP Adapter 库
- 代理可以使用待办事项系统进行复杂的任务规划和管理
- 子代理系统允许将复杂任务分解给专门的子代理，保持主代理上下文清洁

## 进一步资源

- [LangGraph 官方文档](https://langchain-ai.github.io/langgraph/)
- [Tavily API 文档](https://docs.tavily.com/)
- [LangChain 中文文档](https://python.langchain.com/docs/get_started/introduction)
