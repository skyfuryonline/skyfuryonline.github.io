---
layout: post
title: "OpenCompass与VLMEvalKit：大模型评估工具调研与实践"
date: 2025-11-18
author: "Your Name" # 请替换为您的名字
tags: [NLP, 评估, OpenCompass, VLMEvalKit, LLM, VLM]
group: nlp-workshop
catalog: true
---

## 引言

随着大模型（LLM）和多模态模型（VLM）的爆发式发展，如何科学、高效地评估模型能力成为了一个关键问题。上海人工智能实验室推出的 **OpenCompass** 和 **VLMEvalKit** 是目前业界领先的开源评估工具。

本篇博客将详细记录我对这两个工具的调研、安装配置、以及在实际模型上的测试过程。

-   [OpenCompass GitHub](https://github.com/open-compass/opencompass)
-   [VLMEvalKit GitHub](https://github.com/open-compass/VLMEvalKit)

---

## 1. OpenCompass 调研与实践

OpenCompass 是一个用于评估大型语言模型（LLM）的综合平台，支持丰富的模型和数据集。

### 1.1 环境搭建

```bash
# (此处记录 OpenCompass 的安装步骤)
conda create -n opencompass python=3.10
conda activate opencompass
# ...
```

### 1.2 数据集准备

(介绍如何下载和配置评估所需的数据集)

### 1.3 评估脚本编写与运行

```python
# (此处粘贴或解析评估脚本的关键配置)
```

### 1.4 结果分析

(展示评估结果，并分析模型在不同能力维度上的表现)

---

## 2. VLMEvalKit 调研与实践

VLMEvalKit 专注于多模态大模型（VLM）的评估，支持 MME, MMBench 等主流榜单。

### 2.1 环境搭建

```bash
# (此处记录 VLMEvalKit 的安装步骤)
```

### 2.2 评测流程

(描述如何配置 config 文件，以及如何启动评测任务)

### 2.3 自定义模型接入

(记录如何将自己的 VLM 模型接入到 VLMEvalKit 框架中进行评估)

---

## 3. 踩坑记录与解决方案

(记录在使用这两个工具过程中遇到的报错、版本冲突等问题，以及对应的解决方法)

*   **问题 1**: ...
*   **问题 2**: ...

---

## 4. 总结

(总结这两个工具的优缺点，以及在实际项目中的适用场景)
