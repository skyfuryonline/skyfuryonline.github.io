---
layout: post
title: "Generative Agents复现与分析"
date: 2025-11-18
author: "LH"
tags: [NLP, Agent, 复现, LLM,]
group: nlp-workshop 
catalog: true
---

## 摘要

![structure](/img/nlp-workshop/generative-agents-analysis/structure.png)

- 项目介绍：包含用于生成式智能体（一种能够模拟逼真人类行为的计算智能体）的核心模拟模块及其游戏环境。以下将详细介绍如何在本地计算机上设置模拟环境，以及如何将模拟结果以演示动画的形式回放。
- 项目参考：
  - [origin-generative_agents](https://github.com/joonspk-research/generative_agents)
  - [origin-Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442)
  - [GenerativeAgentsCN](https://github.com/x-glacier/GenerativeAgentsCN)
  - [我让六个AI合租，居然出了个海王？](https://www.bilibili.com/video/BV1MkxeeYEEb)
- 主要目标：
  - [x] 理解项目代码及运行原理
  - [x] 修改为vLLM支持并设置本地的LLMs和embedding 模型
  - [x] [相关复现代码](https://github.com/skyfuryonline/GenerativeAgentsCN_remix) 

---

## 运行环境配置及vLLM支持

- 拷贝项目并创建环境
```bash
git clone https://github.com/x-glacier/GenerativeAgentsCN.git

cd GenerativeAgentsCN

conda create -n generative_agents_cn python=3.12
conda activate generative_agents_cn

pip install -U uv
uv pip install -r requirements.txt
```

- 启动运行（默认项目启动方式）
```bash
cd generative_agents
python start.py --name sim-test --start "20250213-09:30" --step 10 --stride 10
```
- 参数说明：
  - name: 每次启动虚拟小镇，需要设定唯一的名称，用于事后回放。
  - start: 虚拟小镇的起始时间。
  - resume: 在运行结束或意外中断后，从上次的“断点”处，继续运行虚拟小镇。
  - step: 在迭代多少步之后停止运行。
  - stride: 每一步迭代在虚拟小镇中对应的时间（分钟）。假如设定--stride 10，虚拟小镇在迭代过程中的时间变化将会是 9:00，9:10，9:20 ...
- 修改`LLMs`和`embedding model`的位置：`generative_agents/data/config.json`
- `replay`: `python replay.py`
  - 默认端口为`5000`，若需要修改，进入`replay`中设置`app.run(debug=True,port=5050)`
  - 可以追加的参数：`name`、`step`、`speed`、`zoom`
- `compress`:`python compress.py --name <simulation-name>`
  - 运行结束后将在`results/compressed/<simulation-name>`目录下生成回放数据文件`movement.json`。同时还将生成`simulation.md`，以时间线方式呈现每个智能体的状态及对话内容

- 启动运行（使用vLLM进行推理）
  - 模型和embedding配置依旧在`generative_agents/data/config.json`
  - 先使用指令启动vLLM服务(`python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B-Instruct`)
  - 进入新的终端启动原始的项目：`cd generative_agents`->`python start.py --name <simulation-name>`
  - 程序现在将连接到本地运行的 VLLM 服务，并使用本地的 Hugging Face 模型进行模拟。后续的 `compress.py` 和 `replay.py` 操作与之前完全相同。
  - 注意：启动的vLLM的模型名与`config.json`中的模型名应一致，如都为：`Qwen/Qwen2.5-7B-Instruct`

---

## 结果展示：

**vLLM配置结果如下（首次启动需要下载模型）：**
![vLLM](/img/nlp-workshop/generative-agents-analysis/vLLM-boot.png)

**再次启动时速度会显著加快：**
![vLLM](/img/nlp-workshop/generative-agents-analysis/vLLM-reboot.png)

**启动项目情况如下（同理启动时需要经过embedding，需要等待一定时间）：**
![process-0](/img/nlp-workshop/generative-agents-analysis/process-0.png)

![process-1](/img/nlp-workshop/generative-agents-analysis/process-1.png)

**结束时的总结输出：**
![final](/img/nlp-workshop/generative-agents-analysis/final.png)

**使用replay回顾过程：**
![replay-1](/img/nlp-workshop/generative-agents-analysis/replay-1.png)

![replay-2](/img/nlp-workshop/generative-agents-analysis/replay-2.png)

---

## 代码分析

## 1. 核心机制解析

(深入剖析Generative Agents的三个核心组成部分)

### 1.1 记忆流 (Memory Stream)

(详细解释记忆流的构成、自然语言转为观测对象的过程、以及如何计算检索得分)

### 1.2 反思 (Reflection)

(阐述Agent如何生成更高层次的抽象思想，以及反思机制的触发条件)

### 1.3 规划 (Planning)

(描述Agent如何制定长期计划，并将其分解为具体的行动步骤)


---

## 总结

(总结本次复现工作的收获，并对Generative Agents的未来发展方向进行展望)
