---
layout: post
title: "博客开发指南 (四)：AI 原生学习教练与长时记忆机制"
subtitle: "如何让大模型拥有上下文连贯性，自动生成月报与周报"
date: 2026-03-24 10:00:00 +0800
author: "skyfury"
header-img: "img/post-bg-universe.jpg"
catalog: true
group: tech
tags:
  - AI
  - Python
  - LLM
  - Prompt Engineering
---

本博客中最硬核的 Python 后端功能，非 `crawlers/weekly_summary_generator.py` 莫属。它不是一个简单调用 API 的脚本，而是一个具备“长时记忆”的 **AI 原生学习教练**。

## 传统 AI 总结的局限

过去，我们把一周的学习日志扔给大模型，让它输出一段总结。但这种方式缺乏连贯性：上周大模型提醒你“图形推理”薄弱，这周它早就忘了。

## 递归记忆机制设计

为了解决这个问题，我重构了生成逻辑：
1. **周报生成**：不仅仅拉取本周所有的 `_gwy_logs`。脚本还会自动寻找 **本月内上一次的周报/月报**。
2. **状态继承**：解析上一份报告的 Front Matter（或者正文结构），将上一次的“教练建议”作为 Context（上下文）注入到本次的 LLM Prompt 中。
3. **跨月压缩**：当时间进入新的一月时，系统会先启动一个任务，把上个月所有的周报组合起来，发送给 LLM 进行高度压缩，生成一份“月度报告（Monthly Report）”。此后，新月的第一周会读取这份“月报”作为上下文底座。

## 集中配置与模型解耦

为了避免把系统提示词（System Prompt）和模型名称硬编码在 Python 代码中，我们使用了 `crawlers/config.json`：
```json
{
  "llm_profiles": {
    "learning_coach": {
      "model_name": "gemini-2.5-flash",
      "system_prompt": "你是一位苛刻且敏锐的学习教练。请着重分析学生的盲区，拒绝空话..."
    }
  }
}
```
结合 LiteLLM 或官方 SDK，这允许我们在不同的提供商（OpenAI, Gemini, DeepSeek）之间无缝切换，仅需修改配置文件。
        