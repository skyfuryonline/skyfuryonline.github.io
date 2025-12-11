---
layout: post
title: "Generative Agents复现"
date: 2025-11-17
author: "LH"
tags: [NLP, Agent, 复现, LLM]
group: nlp-workshop
catalog: true
---

## 引言 
项目相关参考如下: 
[generative_agents](https://github.com/joonspk-research/generative_agents)  
[Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442)  
[Generative-agent-proj](https://github.com/skyfuryonline/Generative-agent-proj) 
[Generative Agents和代码实践](https://blog.csdn.net/qq_35812205/article/details/143484516)  
[我让六个AI合租，居然出了个海王?](https://www.bilibili.com/video/BV1MkxeeYEEb/)  
![project-cover](/img/nlp-workshop/generative-agents/cover.png) 

- 包含用于生成式智能体（一种能够模拟逼真人类行为的计算智能体）的核心模拟模块及其游戏环境；
- 使用`memory->reflection->planning`框架提高NPC的行为目标性；
  - 为每个AI编写人设：例如，John Lin是Willow市场和药房的药店店员，他喜欢帮助别人。他总是想方设法使顾客买药的过程更容易。
  - 创建可互动的小镇：智能体将会在这个被称为SmallVille的小镇里生活。研究人员为小镇配置了很多可以互动的组件，有居民区，Cafe，酒吧，大学，商店，公园。
  - 为每个AI创建记忆流管理系统：智能体使用自然语言储存与它相关的完整记录，将这些记忆随着时间的推移合成为更高层次的思考，并动态地检索它们来规划行为。
- 详细记录如何在本地计算机上设置模拟环境，以及如何将模拟结果以演示动画的形式回放；

![generative-agent-architecture](/img/nlp-workshop/generative-agents/architecture.png)

---

## 环境搭建

**初始化项目和对应的配置环境：**    
```bash
git clone https://github.com/joonspk-research/generative_agents.git 
conda create -n generative_agent python=3.11 -y   
conda activate generative_agent 
# 使用uv进行包管理  
pip install -U uv   
uv pip install -r ./requirements.txt    
```

**在`reverie/backend_server`新建utils.py：**
```python
# reverie/backend_server/utils.py   
# Copy and paste your OpenAI API Key    
openai_api_key = "<Your OpenAI API>"
# Put your name 
key_owner = "<Name>"

maze_assets_loc = "../../environment/frontend_server/static_dirs/assets"
env_matrix = f"{maze_assets_loc}/the_ville/matrix"
env_visuals = f"{maze_assets_loc}/the_ville/visuals"

fs_storage = "../../environment/frontend_server/storage"
fs_temp_storage = "../../environment/frontend_server/temp_storage"

collision_block_id = "32125"

# Verbose   
debug = True
```

> 考虑到服务器具有一定的计算资源，本次部署将修改为使用本地模型进行拟真。模型选用`Qwen2.5-7B-Instruct`,嵌入模型选用`all-MiniLM-L6-v2`,同时使用`vLLM`加速推理；







## 数据与模拟环境

(说明模拟环境中Agent的初始化、交互逻辑和数据收集方式)

## 代码分析

(对复现代码的关键部分进行说明，例如：记忆流、反思机制、规划过程等)

---

## 实验结果与分析

(展示复现的实验结果，与原论文结果进行对比，分析异同)

![Generative Agents 模拟场景示例](/img/nlp-workshop/generative-agents/simulation-example.png)


---

## 总结

(总结本次复现工作的收获、挑战和对Agent领域发展的思考)
