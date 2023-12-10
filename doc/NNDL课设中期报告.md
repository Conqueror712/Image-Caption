## 题目：基于编解码框架方法的图像描述生成

> 2023 秋季北京邮电大学深度学习与神经网络课程设计

目录：

[TOC]



## 一、详细设计

### 1.1 系统架构

#### 1.1.1 Transformer Model 系统架构

我们首先使用 argparse 库解析命令行参数，获取图像路径、模型版本和 Checkpoint 路径；其次根据命令行参数加载预训练模型，或者从 Checkpoint 加载模型（可选）；紧接着使用 PIL 库打开图像，并进行预处理；然后使用模型生成图像的描述；最后使用 METEOR 和 ROUGE-L 评估生成的描述与参考描述的相似度。

> <img src="../doc/img/Transformer_framework.png" alt="image" style="zoom: 50%;" />
>
> 图1：Transformer Model 系统架构图

#### 1.1.2 Attention Model 系统架构

> 暂略...

### 1.2 模块划分

#### 1.2.1 Transformer Model 模块划分

1. **参数解析模块**：解析命令行参数，获取图像路径、模型版本和 Checkpoint 路径。

    ```Python
    parser = argparse.ArgumentParser(description='Image Captioning')
    parser.add_argument('--path', type=str, help='path to image', required=True)
    parser.add_argument('--v', type=str, help='version')
    parser.add_argument('--checkpoint', type=str, help='checkpoint path', default=None)
    args = parser.parse_args()
    image_path = args.path
    version = args.v
    checkpoint_path = args.checkpoint
    ```

2. **模型加载模块**：根据命令行参数加载预训练模型，或者从 Checkpoint 加载模型。

3. **图像预处理模块**：使用 PIL 库打开图像，并进行预处理。

    ```Python
    start_token = tokenizer.convert_tokens_to_ids(tokenizer._cls_token)
    end_token = tokenizer.convert_tokens_to_ids(tokenizer._sep_token)
    image = Image.open(image_path)
    image = coco.val_transform(image)
    image = image.unsqueeze(0)
    ```

4. **Caption生成模块**：使用模型生成图像的描述。

    ```Python
    def create_caption_and_mask(start_token, max_length):
        caption_template = torch.zeros((1, max_length), dtype=torch.long)
        mask_template = torch.ones((1, max_length), dtype=torch.bool)
    
        caption_template[:, 0] = start_token
        mask_template[:, 0] = False
    
        return caption_template, mask_template
    
    caption, cap_mask = create_caption_and_mask(start_token, config.max_position_embeddings)
    
    def evaluate():
        model.eval()
        for i in range(config.max_position_embeddings - 1):
            predictions = model(image, caption, cap_mask)
            predictions = predictions[:, i, :]
            predicted_id = torch.argmax(predictions, axis=-1)
    
            if predicted_id[0] == 102:
                return caption
    
            caption[:, i+1] = predicted_id[0]
            cap_mask[:, i+1] = False
    
        return caption
    ```

5. **评估模块**：使用 METEOR 和 ROUGE-L 评估生成的描述与参考描述的相似度。

    ```Python
    def calc_meteor(reference, hypothesis):
        hypothesis = word_tokenize(hypothesis)
        reference = word_tokenize(reference)
        return single_meteor_score(reference, hypothesis)
    
    
    def calc_rouge_l(reference, hypothesis):
        rouge = Rouge()
        scores = rouge.get_scores(hypothesis, reference)
        return scores[0]['rouge-l']['f']
    ```

    

#### 1.2.2 Attention Model 模块划分

> 暂略...

### 1.3 接口设计

#### 1.3.1 Transformer Model 接口设计

1. `argparse.ArgumentParser`：用于解析命令行参数。
2. `torch.hub.load`：用于加载预训练模型。
3. `PIL.Image.open`：用于打开图像。
4. `nltk.translate.meteor_score.single_meteor_score`：用于计算 METEOR 分数。
5. `rouge.Rouge.get_scores`：用于计算 ROUGE-L 分数。

#### 1.3.2 Attention Model 接口设计

> 暂略...

### 1.4 技术方案

#### 1.4.1 Transformer Model 技术方案

1. **模型**：本程序使用了基于 Transformer 的编解码模型，具体来说，是使用了 BERT 作为编码器，用于生成图像的描述。模型可以从预训练模型加载，也可以从 Checkpoint 加载。
2. **图像预处理**：图像预处理主要包括打开图像和进行变换。变换主要是使用了 COCO 数据集的验证集变换。
3. **Caption生成**：Caption 生成主要是通过模型生成图像的描述。首先，创建一个 caption 和一个 mask，然后在每一步中，使用模型预测下一个词，直到预测出结束标记。
4. **评估**：评估主要是使用 METEOR 和 ROUGE-L 评估生成的描述与参考描述的相似度。METEOR 分数是基于单词级别的评估，而 ROUGE-L 分数是基于句子级别的评估。

#### 1.4.2 Attention Model 技术方案

> 暂略...

## 二、已完成工作

- ✅Transformer Model 的初步编写和测试
- ✅METEOR 和 ROUGE-L 评估指标函数的编写和测试

## 三、初步结论

### 3.1 Transformer Model 结论

1. **模型性能**：通过 METEOR 和 ROUGE-L 评分，我们可以看到目前的评分较低，需要进一步优化模型或调整参数。
2. **模型泛化能力**：实验表明，模型在各种类型的图像上都能生成质量尚可的描述，我们可以初步得出结论，即该模型具有良好的泛化能力。
3. **模型运行效率**：经过测试，模型可以在合理的时间内生成描述，并且所需的计算资源较少（Windows OS + GTX1650 描述一张图片大约需要 10 秒），模型在效率方面表现尚可。

### 3.2 Attention Model 结论

> 暂略...

## 四、问题及可能的解决方案

### 4.1 共性问题

目前的 `train_captions.json` 文件中，每张图片的关键点是包含在图片名称里的，然而，如果将其分出来，作为单独的一个属性的话会好处理一些，这是可以优化的地方，代码已经放在 `./Model2_Transformer/data_preprocessing` 了，该问题已处理。

### 4.2 Transformer Model 问题

1. **模型性能**：目前的评分较低，需要进一步优化模型或调整参数。
2. **模型泛化能力**：模型在各种类型的图像上都能生成质量尚可的描述，但是对于本实验的目标——服装图像描述，并没有得到符合预期的描述结果，即针对性不强。

基于此，我们可以给出解决方案：在给定的服装数据集上进行微调。然而，由于需要在训练时关注需要关注的点，所以这还并不只是简单的通用训练流程就可以解决的问题，还需要进一步的探索与思考。

### 4.3 Attention Model 问题

> 暂略...

## 五、后续工作计划

- 对于 Transformer Model 来说，下一阶段需要进行在给定的服装数据集上进行微调。
- 对于可选任务来说，需要调研一下多模态模型如何帮助图像描述任务。