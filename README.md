## 零、前言

> 任务介绍：

自动为图片生成流畅关联的自然语言描述，例如：

![image](/img/01.png)

---

> 我们的选择：

- 数据集：仅用到 image 和 textual descriptions，下载链接见下方
    - 请注意文本描述不是单一句子 
    - 训练 / 测试集切分会在下周发到课程 QQ 群
- 我们选择的两种模型结构：
    - 网格 / 区域表示、自注意力 + 注意力 
    - 网格 / 区域表示、Transformer 编码器 + Transformer 解码器 
- 我们选择的三种评测标准：
    - METEOR
    - ROUGE-L
    - CIDEr-D

> 选做要求：

- 默认实现的交叉熵损失和评测指标不一致，请实现基于强化学习的损失函数，直接优化评测指标 
- 微调多模态预训练模型或多模态大语言模型，并测试性能
- 利用训练的服饰图像描述模型和多模态大语言模型，为真实背景的服饰图像数据集增加服饰描述和背景描述，构建全新的服饰图像描述数据集 
    - 在新数据集上重新训练服饰图像描述模型

## 一、快速开始

> 环境检查：
>
> - Linux Ubuntu 20.04 / 22.04
> - NVIDIA GPU
> - CUDA 12.2

1. 克隆该项目：

```
git clone git@github.com:Conqueror712/Image-Description.git
```

2. 进入根目录：

```
cd Image-Description
```

3. 下载数据集，放入根目录：

```
# images:
https://drive.google.com/file/d/1U2PljA7NE57jcSSzPs21ZurdIPXdYZtN/view

# textual descriptions:
https://drive.google.com/file/d/1d1TRm8UMcQhZCb6HpPo8l3OPEin4Ztk2/view
```

4. 数据集切分：

```
...
```


5. 运行 demo：

```
python main.py
```

6. 查看结果：

```
...
```

## 二、方法介绍

首先看一下目录结构：

```
.
|-- main.py		# 运行脚本
```

