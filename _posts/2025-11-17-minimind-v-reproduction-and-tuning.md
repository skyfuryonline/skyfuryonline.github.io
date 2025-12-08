---
layout: post
title: "Minimind-v复现及调优"
date: 2025-11-17
author: "LH"
tags: [LLM, Minimind, 复现, 预训练, 后训练]
group: llm
catalog: true
---

## 引言

![logo](/img/llm/minimind/logo.png)

[Minimind-v项目链接](https://github.com/jingyaogong/minimind-v)

- minimind旨在从0开始，仅用1.3块钱成本 + 1小时！即可训练出26M参数的超小多模态视觉语言模型MiniMind-V。
- MiniMind-V最小版本体积仅为 GPT3 的约 $\frac{1}{7000}$，力求做到个人GPU也可快速推理甚至训练。
- MiniMind-V是MiniMind纯语言模型的视觉能力额外拓展。
- 项目同时包含了VLM大模型的极简结构、数据集清洗、预训练(Pretrain)、监督微调(SFT)等全过程代码。
- 这不仅是一个开源VLM模型的最小实现，也是入门视觉语言模型的简明教程。
- 希望此项目能为所有人提供一个抛砖引玉的示例，一起感受创造的乐趣！推动更广泛AI社区的进步！

---

## 复现过程

### 1. 环境配置

克隆项目及模型：
```bash
# clone项目文件
git clone https://github.com/jingyaogong/minimind-v.git

# 下载clip模型到 ./model/vision_model 目录下
hf download  openai/clip-vit-base-patch16 --local-dir /home/lihao/minimind-v/model/vision_model/clip-vit-base-patch16

# 下载minimind语言模型权重到 ./out 目录下（作为训练VLM的基座语言模型）
# HuggingFace
https://huggingface.co/jingyaogong/MiniMind2-V-PyTorch/blob/main/llm_768.pth # or llm_512.pth
```

![model_weight](/img/llm/minimind/model_weight.png)

创建环境：
```bash
conda create -n minimind python=3.11 -y

conda activate minimind

# 使用uv进行包管理
pip install -U uv
uv pip install -r requirements.txt 
```

测试已有模型的效果（使用项目自带的eval_vlm.py）：
```bash
# 指令的参数解析见eval_vlm.py中main部分
python eval_vlm.py --load_from model --hidden_size 768 --weight llm
```

效果如下图：
![origin](/img/llm/minimind/before_PT.png)


### 2. 数据准备

```bash
# pretrain阶段的数据集
cd ./dataset
wget https://hf-mirror.com/datasets/jingyaogong/minimind-v_dataset/resolve/main/pretrain_data.jsonl
wget https://hf-mirror.com/datasets/jingyaogong/minimind-v_dataset/resolve/main/pretrain_images.zip
unzip pretrain_images.zip && rm pretrain_images.zip

# SFT阶段的数据集
cd ./dataset
wget https://hf-mirror.com/datasets/jingyaogong/minimind-v_dataset/resolve/main/sft_data.jsonl
wget https://hf-mirror.com/datasets/jingyaogong/minimind-v_dataset/resolve/main/sft_images.zip
unzip sft_images.zip && rm sft_images.zip
```

### 3. 模型训练

pretrained训练(学习图像描述)：

```bash
# 注意这步必不可少，否则后续加载tokenizer的路径会有误
cd trainer/

# 基础训练命令（从LLM权重开始，仅训练vision_proj）
python train_pretrain_vlm.py --epochs 4 --from_weight llm --hidden_size 768

# 使用多张GPU进行DDP：
# 设置tmux进行后台训练
tmux new-session -t mm
torchrun --nproc_per_node 2 train_pretrain_vlm.py --epochs 4 --from_weight llm --hidden_size 768 --batch_size 256
```

![PT效果图](/img/llm/minimind/PT.png)

![PT_GPU_usage](/img/llm/minimind/PT_GPU_usage.png)


测试PT模型的效果（使用项目自带的eval_vlm.py）：
```bash
# 指令的参数解析见eval_vlm.py中main部分
python eval_vlm.py --load_from model --hidden_size 768 --weight pretrain_vlm
```

效果如图：

![after_PT](/img/llm/minimind/after_PT.png)


SFT训练（学习看图对话）：

```bash
cd trainer/

# 基础训练命令（从预训练权重开始，全参数微调）
python train_sft_vlm.py --epochs 2 --from_weight pretrain_vlm --hidden_size 768 --batch_size 256

# 使用多张GPU进行DDP：
torchrun --nproc_per_node 2 train_sft_vlm.py --epochs 2 --from_weight pretrain_vlm --hidden_size 768 --batch_size 256
```

测试SFT模型的效果（使用项目自带的eval_vlm.py）：
```bash
# 指令的参数解析见eval_vlm.py中main部分
python eval_vlm.py --load_from model --hidden_size 768 --weight sft_vlm
```

效果如图：

![after_SFT](/img/llm/minimind/after_SFT.png)

---

## 训练代码及模型细节分析

### 模型结构如下：

![minimind模型细节-DenseModel](/img/llm/minimind/VLM-structure.png)

### 模型输入示例：

VLM的输入依然是一段文本，其中包含特殊的`<image>`占位符。 在计算文本嵌入后，可以将图像编码器生成的向量投影到该占位符对应的嵌入部分，替换掉原先的占位符embedding。 例如：

```text
<image>\n这个图像中有什么内容？
```

在minimind-v中，使用196个字符组成的 `@@@...@@@` 占位符代替图像，之所以是196个字符，前面有所提及： 任何图像都被clip模型encoder为196×768维的token， 因此minimind-v的prompt为：

```text
@@@......@@@\n这个图片描述的是什么内容？
```

计算完embedding和projection，并对图像部分token替换后整个计算过程到输出则和LLM部分没有任何区别。

![minimind-input](/img/llm/minimind/minimind-v-input.png)

一次性多图的实现方法就是通过注入多个`<image>`图像占位符进行实现，不需要修改任何框架。

### 视觉模块分析

[openai/clip-vit-base-patch16](https://huggingface.co/openai/clip-vit-base-patch16)



### 训练代码分析


---

## 总结

