## Image-Caption: 基于编解码框架的图像描述

> 2023 秋季北京邮电大学深度学习与神经网络课程设计

## 一、项目目录结构介绍

```
Image-Caption/
|-- data_new/						# 新版数据
|   |-- output/						# 模型1使用新版数据生成的输出结果
|   |-- test_images/				# 新版数据的测试集
|   |-- train_images_1/				# 新版数据的训练集（第一部分）
|   |-- train_images_2/				# 新版数据的训练集（第二部分）
|	|-- rename_script.py			# 文件重命名脚本
|   |-- BLIP_test_captions.json		# 多模态模型生成的测试集的图像描述文件
|   |-- BLIP_train_captions.json	# 多模态模型生成的训练集的图像描述文件
|	|-- Model2_test_captions.json	# 模型2生成的测试集的图像描述文件
|	|-- Model2_train_captions_1.json# 模型2生成的训练集的图像描述文件（第一部分）
|-- data_old/						# 旧版数据
|   |-- output/						# 模型1使用旧版数据生成的输出结果
|   |-- test_images/				# 旧版数据的测试集
|   |-- train_images/				# 旧版数据的训练集
|	|-- label.json					# 加入关键点后的全量json数据
|   |-- test_captions.json			# 原始给定的测试集的图像描述文件
|   |-- train_captions.json			# 原始给定的训练集的图像描述文件
|-- doc/							# 项目的需求文档及项目报告
|-- Ex1_BLIP						# 附加任务1：多模态模型
|	|-- Salesforce/					# 模型文件
|	|-- run_fulldata_script.py		# 全量数据运行脚本
|	|-- run_script.py				# 单个数据运行脚本
|-- Ex2_RL_Loss						# 附加任务2：基于强化学习的损失函数
|-- Model1_YellowOrange 			# 模型1：Self-Attention + Attention模型
|-- Model2_Transformer  			# 模型2：Transformer Encoder + Decoder模型
|-- Original_Model					# 模型0：初始模型的图像描述模型
|-- .gitignore
|-- LICENSE
|-- README.md						# 项目的简介
```


## 二、小组分工与时间安排

|                            巩羽飞                            |                       黄成梓                       |
| :----------------------------------------------------------: | :------------------------------------------------: |
| 模型：网格 / 区域表示、Transformer 编码器 + Transformer 解码器 |      模型：网格 / 区域表示、自注意力 + 注意力      |
|                    指标：METEOR + ROUGE-L                    |                   指标：CIDEr-D                    |
|   其他：微调多模态预训练模型或多模态大语言模型，并测试性能   | 其他：实现基于强化学习的损失函数，直接优化评测指标 |

|   11.25   |        11.30         |             12.12              |                            12.28                             |
| :-------: | :------------------: | :----------------------------: | :----------------------------------------------------------: |
| 开题报告✅ | 模型跑通 + 评测指标✅ | 中期报告 + 附加任务: 优化指标✅ | 结题报告 + 附加任务: 微调多模态大模型 + 附加任务: 强化学习损失函数✅ |
