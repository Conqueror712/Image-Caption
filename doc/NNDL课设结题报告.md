## 结题报告——Image Caption

> 2023 秋季北京邮电大学深度学习与神经网络课程设计
>
> 【注：后续会挪到 Jupyter Notebook 中】

目录：

[TOC]

## 一、任务说明

我们开发了一个基于编解码框架的图像描述生成系统。这个系统能够自动为输入的图片生成流畅且关联的自然语言描述。我们采用了 Show and Tell, Attention and Self-Attention, Transformer Encoder and Decoder 这三个模型结构来实现这个任务，接下来我们将在第四部分逐个介绍。

至于什么是图像描述技术，其实就是以图像为输入，通过数学模型和计算使计算机输出对应图像的自然语言描述文字，使计算机拥有看图说话的能力，是图像处理领域中继图像识别、图像分割和目标跟踪之后的又一新型任务。在日常生活中，人们可以将图像中的场景、色彩、逻辑关系等低层视觉特征信息自动建立关系，从而感知图像的高层语义信息，但是计算机作为工具只能提取到数字图像的低层数据特征，而无法像人类大脑一样生成高层语义信息，这就是计算机视觉中的语义鸿沟问题。图像描述技术的本质就是将计算机提取的图像视觉特征转化为高层语义信息，即解决语义鸿沟问题，使计算机生成与人类大脑理解相近的对图像的文字描述，从而可以对图像进行分类、检索、分析等处理任务。

我们通过完成这个课程设计作业，深入理解了编解码框架、自注意力机制、Transformer 模型等先进的深度学习技术，并能够将这些技术应用到实际问题中，不仅让我们更好的掌握了理论课上学到的知识，更锻炼了我们的动手实践能力。

## 二、实验数据

### 2.1 原始数据

我们使用了 DeepFashion-MultiModal 数据集中 image 和 textual descriptions 的数据，其中 80% 的数据作为模型的训练集，20% 作为模型的测试集。数据集的 Github Repo 如下：

>  https://github.com/yumingj/DeepFashion-MultiModal

由于数据对应的 json 文件已经提前划分好，但是 images 文件夹仍然是混合在一起的，所以我们编写了一个 Python 脚本用于将 images 划分为 train_images 和 test_images，如下所示：

```Python
import os
import json
import shutil

# 读取json文件并转换为字典
with open('../../data/test_captions.json', 'r') as f:
    test_captions = json.load(f)

with open('../../data/train_captions.json', 'r') as f:
    train_captions = json.load(f)

# 指定源目录和目标目录
source_directory = '../../data/images'
train_directory = '../../data/train_images'
test_directory = '../../data/test_images'

# 确保目标目录存在
os.makedirs(train_directory, exist_ok=True)
os.makedirs(test_directory, exist_ok=True)

# 将训练集图片复制到目标目录
for image in train_captions:
    shutil.copy(os.path.join(source_directory, image), train_directory)

# 将测试集图片复制到目标目录
for image in test_captions:
    shutil.copy(os.path.join(source_directory, image), test_directory)
```

另外，我们还发现图像的关键点信息并没有在 json 文件中显示，而是在图像的文件名中，所以我们通过正则表达式，提取了每张图像的关键点信息，并更新了 json 文件，如下所示：

```Python
import os
import re
import json

# 定义一个函数来解析文件名
def parse_filename(filename):
    # 使用正则表达式匹配文件名
    pattern = r'^(?P<gender>\w+)-(?P<clothing>[\w_]+)-id_(?P<id>\d+)-(?P<group>\d+)_(\d+_(?P<body>\w+))\.jpg$'
    match = re.match(pattern, filename)
    if match:
        return match.groupdict()
    else:
        return None

# 定义一个函数来处理目录中的所有文件
def process_directory(directory):
    # 创建一个字典来存储结果
    results = {}
    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        # 解析文件名
        info = parse_filename(filename)
        if info:
            # 将解析的信息与文件名关联起来
            results[filename] = info
    return results

# 使用函数处理目录
directory = '../../data/images'
results = process_directory(directory)

# 将结果保存到json文件中
with open('../../data/label.json', 'w') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)
```

### 2.2 背景描述增量数据

> 暂无，待找。

## 三、实验环境

- Ubuntu 20.04 / Ubuntu 22.04 / Windows 11
- NVIDIA GPU and NVIDIA CUDA Driver
- CUDA 12.2 / CUDA 11.8
- Python 3.10

具体的第三方库依赖请见 `requirements.txt`

## 四、模型选择

首先，作为共性特点，我们先来介绍一下基于编解码器的方法：

随着深度学习技术的不断发展，神经网络在计算机视觉和自然语言处理领域得到了广泛应用。受机器翻译领域中编解码器模型的启发，图像描述可以通过端到端的学习方法直接实现图像和描述句子之间的映射，将图像描述过程转化成为图像到描述的"翻译"过程。基于深度学习的图像描述生成方法大多采用以 CNN-RNN 为基本模型的编解码器框架，CNN 决定了整个模型的图像识别能力，其最后的隐藏层的输出被用作解码器的输入，RNN 是用来读取编码后的图像并生成文本描述的网络模型，下图是一个简单递归神经网络 RNN 和多模态递归神经网络 m-RNN 架构的示意图：

![image](../doc/img/01.png)

之所以是 CNN 决定了整个模型的图像识别能力，RNN 被用来读取编码后的图像并生成文本描述的网络模型，是因为计算机视觉问题有很强的局部特征，即一个像素和它周围的旁边几个像素有很强的关联性，但是离他非常远的像素之间的关联性就比较弱，所以只需要像 CNN 一样在局部做连接，而不需要像全连接网络一样每一层之间都是全连接的，从而大大降低了权重的数量。而 RNN 通过使用带自反馈的神经元，也就是隐藏状态，能够处理任意长度的序列数据，可以有效保存序列数据的历史信息。

![image](../doc/img/CNN.png)

![image](../doc/img/RNN.png)

### 4.1 Show and Tell（就是老师给的那个）

### 4.2 Attention and Self Attention

近年来，注意力机制被广泛应用于计算机视觉领域，其本质是为了解决编解码器在处理固定长度向量时的局限性。注意力机制并不是将输入序列编码成一个固定向量，而是通过增加一个上下文向量来对每个时间步的输入进行解码，以增强图像区域和单词的相关性，从而获取更多的图像语义细节，下图是一个学习单词 / 图像对齐过程的示意图：

![image](../doc/img/02.png)

我们利用网格 / 区域表示 + 自注意力 + 注意力的模型结构来完成副使图像描述任务。

#### 4.2.1 参数配置方面：

`configurations.py` 文件中定义的 `Config` 类作为项目的配置中心，其作用是集中管理项目中使用的所有配置参数。这些参数通常包括文件路径、模型参数、数据处理选项、训练设置和图像处理参数等。通过这种方式，可以在不修改代码的情况下调整项目的行为。以下是 `Config` 类中定义的配置参数及其作用：

1. **数据路径**：
   - `data_path`：主数据目录路径。
   - `images_path`：存储图像的路径。
   - `train_captions_path`：训练集的文本描述文件路径。
   - `test_captions_path`：测试集的文本描述文件路径。
   - `output_folder`：用于存储词汇表和处理后数据的输出文件夹路径。

2. **模型参数**：
   - `embed_size`：嵌入向量的维度。
   - `vocab_size`：词汇表的大小。
   - `num_layers`：定义循环神经网络中的层数。
   - `num_heads`：自注意力机制中头的数量。
   - `dropout`：在模型中使用的 Dropout 比率。
   - `hidden_size`：隐藏层的维度。
   - `image_code_dim`：图像编码的维度。
   - `word_dim`：词嵌入的维度。
   - `attention_dim`：注意力机制的隐藏层维度。

3. **数据处理参数**：
   - `min_word_count`：词汇表中词的最小出现次数，用于筛选词汇。
   - `max_len`：假设的描述的最大长度。

4. **训练参数**：
   - `batch_size`：每个批次的大小。
   - `learning_rate`：学习率。
   - `num_epochs`：训练的总轮次数。
   - `workers`：加载数据时使用的工作线程数。
   - `encoder_learning_rate`：编码器的学习率。
   - `decoder_learning_rate`：解码器的学习率。
   - `lr_update`：学习率更新频率。

5. **图像预处理参数**：
   - `image_size`：图像缩放后的大小。
   - `crop_size`：从缩放后的图像中裁剪出的大小。

6. **其他配置**：
   - `device`：设置运行计算的设备，如果 CUDA 可用则使用 GPU，否则使用 CPU。

#### 4.2.2 数据预处理方面：

为图像描述任务准备和预处理数据，确保数据能够被模型以适当的格式接受和处理，数据预处理是建立有效的训练和测试环境的基础。我们实现了 `datasets.py` 文件来进行数据预处理，以下是主要功能：

1. `create_dataset` 函数：用于处理原始文本描述，创建一个词汇表，并将文本转换为对应的词索引向量。它首先读取训练和测试数据集中的文本描述，然后统计词频以创建词汇表，并移除低频词。之后，它定义了一个内部函数 `encode_captions`，这个函数负责将每条文本描述转换为一个固定长度的词索引序列，包括特殊标记 <start>, <end>, <pad>, 和 <unk>。转换完成后，函数将这些数据保存为 json 文件，以便后续处理。部分代码展示如下：

    ```Python
    def create_dataset(max_len=64):
        """
        整理数据集，构建词汇表，并将文本描述转换为词索引向量。
        使用configuration.py文件中定义的配置信息。
        """
        # 使用config中定义的路径
        image_folder = config.images_path
        train_captions_path = config.train_captions_path
        test_captions_path = config.test_captions_path
        output_folder = config.output_folder
    
        # 读取训练图像描述
        with open(train_captions_path, 'r') as f:
            train_captions_data = json.load(f)
    
        # 读取测试图像描述
        with open(test_captions_path, 'r') as f:
            test_captions_data = json.load(f)
    
        # 统计训练集的文本描述的词频
        vocab = Counter()
        for caption in train_captions_data.values():
            vocab.update(caption.lower().split())
    
        # 移除其中的低频词
        vocab = {word for word, count in vocab.items() if count >= config.min_word_count}
    
        # 构建词典
        word_to_idx = {word: idx + 4 for idx, word in enumerate(vocab)}
        word_to_idx['<pad>'] = 0
        word_to_idx['<start>'] = 1
        word_to_idx['<end>'] = 2
        word_to_idx['<unk>'] = 3
    
        # 一个函数来转换描述为词索引向量，并进行填充
        def encode_captions(captions_data, word_to_idx, max_len):
            encoded_captions = {}
            caplens = {}
            for img_id, caption in captions_data.items():
                words = caption.lower().split()
                encoded_caption = [word_to_idx.get(word, word_to_idx['<unk>']) for word in words]
                caplen = len(encoded_caption) + 2  # 加2是因为还要加上<start>和<end>
                encoded_caption = [word_to_idx['<start>']] + encoded_caption + [word_to_idx['<end>']]
                encoded_caption += [word_to_idx['<pad>']] * (max_len - len(encoded_caption))
                encoded_captions[img_id] = encoded_caption[:max_len]
                caplens[img_id] = caplen if caplen <= max_len else max_len
            return encoded_captions, caplens
    
        # 对训练集描述进行编码
        encoded_captions_train, caplens_train = encode_captions(train_captions_data, word_to_idx, max_len)
    
        # 对测试集描述进行编码
        encoded_captions_test, caplens_test = encode_captions(test_captions_data, word_to_idx, max_len)
    
        # 存储词典和编码后的描述
        with open(os.path.join(output_folder, 'vocab.json'), 'w') as f:
            json.dump(word_to_idx, f)
    
        with open(os.path.join(output_folder, 'encoded_captions_train.json'), 'w') as f:
            json.dump(encoded_captions_train, f)
    
        with open(os.path.join(output_folder, 'encoded_captions_test.json'), 'w') as f:
            json.dump(encoded_captions_test, f)
    
        # 存储图像路径
        image_paths_train = {img_id: os.path.join(image_folder, img_id) for img_id in train_captions_data.keys()}
        with open(os.path.join(output_folder, 'image_paths_train.json'), 'w') as f:
            json.dump(image_paths_train, f)
    
        image_paths_test = {img_id: os.path.join(image_folder, img_id) for img_id in test_captions_data.keys()}
        with open(os.path.join(output_folder, 'image_paths_test.json'), 'w') as f:
            json.dump(image_paths_test, f)
    
        # 存储caplens
        with open(os.path.join(output_folder, 'caplens_train.json'), 'w') as f:
            json.dump(caplens_train, f)
    
        with open(os.path.join(output_folder, 'caplens_test.json'), 'w') as f:
            json.dump(caplens_test, f)
    ```

2. `ImageTextDataset` 类：继承自 `torch.utils.data.Dataset`，这个类是一个 PyTorch 的自定义数据集，用于加载图像和对应的已编码文本描述。它重写了 `__getitem__` 方法，用于获取索引对应的数据点（图像和文本描述），并将图像通过转换处理成统一的格式；重写了 `__len__` 方法，返回数据集的大小，部分代码展示如下：

    ```Python
    class ImageTextDataset(Dataset):
        """
        PyTorch数据集类，用于加载和处理图像-文本数据。
        """
    
        def __init__(self, image_paths_file, captions_file, caplens_file, transform=None):
            """
            初始化数据集类。
            参数:
                image_paths_file: 包含图像路径的json文件路径。
                captions_file: 包含编码后文本描述的json文件路径。
                transform: 应用于图像的预处理转换。
            """
            # 载入图像路径和文本描述以及caplens
            with open(image_paths_file, 'r') as f:
                self.image_paths = json.load(f)
    
            with open(captions_file, 'r') as f:
                self.captions = json.load(f)
    
            with open(caplens_file, 'r') as f:
                self.caplens = json.load(f)
    
            # 设置图像预处理方法
            self.transform = transform or transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
        def __getitem__(self, index):
            """
            获取单个数据点。
            参数:
                index: 数据点的索引。
            返回:
                一个包含图像和对应文本描述的元组。
            """
            # 获取图像路径和文本描述以及caplen
            image_id = list(self.image_paths.keys())[index]
            image_path = self.image_paths[image_id]
            caption = self.captions[image_id]
            caplen = self.caplens[image_id]
    
            # 加载图像并应用预处理
            image = Image.open(image_path).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
    
            # 将文本描述转换为张量
            caption_tensor = torch.tensor(caption, dtype=torch.long)
    
            return image, caption_tensor, caplen
    
        def __len__(self):
            """
            数据集中的数据点总数。
            """
            return len(self.image_paths)
    ```

3. `create_dataloaders` 函数：使用 `ImageTextDataset` 类来创建PyTorch的 `DataLoader`，它提供了一个可迭代的数据加载器，用于在训练和测试时批量加载数据，并可选地对数据进行打乱和多进程加载，部分代码展示如下：

    ```Python
    # 创建训练集和测试集的 DataLoader
    def create_dataloaders(config):
        """
        创建训练集和测试集的 DataLoader。
    
        参数:
            batch_size: 每个批次的大小。
            num_workers: 加载数据时使用的进程数。
            shuffle_train: 是否打乱训练数据。
    
        返回:
            train_loader: 训练数据的 DataLoader。
            test_loader: 测试数据的 DataLoader。
        """
        # 图像预处理转换
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
        # 加载数据时使用的进程数
        num_workers = 0
    
        # 创建数据集对象
        train_dataset = ImageTextDataset(
            image_paths_file=os.path.join(config.output_folder, 'image_paths_train.json'),
            captions_file=os.path.join(config.output_folder, 'encoded_captions_train.json'),
            caplens_file=os.path.join(config.output_folder, 'caplens_train.json'),
            transform=transform
        )
    
        test_dataset = ImageTextDataset(
            image_paths_file=os.path.join(config.output_folder, 'image_paths_test.json'),
            captions_file=os.path.join(config.output_folder, 'encoded_captions_test.json'),
            caplens_file=os.path.join(config.output_folder, 'caplens_test.json'),
            transform=transform
        )
    
        # 创建 DataLoader 对象
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
    
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=config.batch_size,
            shuffle=False,  # 通常测试集不需要打乱
            num_workers=num_workers,
            pin_memory=True
        )
    
        return train_loader, test_loader
    ```

4. 我还定义了一个 `datasets_pretrain_demo.py` 文件来验证数据预处理过程是否正确。它通过以下步骤实现这一目标：

    - 读取词汇表和编码后的描述：加载之前生成的词汇表 `vocab.json`，编码后的训练集描述 `encoded_captions_train.json`，以及训练图像的路径 `image_paths_train.json`。
    - 索引到单词的转换： 创建从词索引到单词的反向映射，用于将编码后的描述转换回文本形式。
    - 选择并展示图像： 从图像路径列表中选择第一个图像 ID，并加载对应的图像。
    - 展示图像：使用 matplotlib 展示图像，并关闭坐标轴。
    - 打印文本描述：将编码后的描述（词索引列表）转换回单词形式，并打印出来，以验证编码和图像加载的正确性。
    
    ```Python
    import json
    from PIL import Image
    import matplotlib.pyplot as plt
    
    vocab_path = '../data/output/vocab.json'
    encoded_captions_path = '../data/output/encoded_captions_train.json'
    image_paths_path = '../data/output/image_paths_train.json'
    
    # 读取词典、编码后的描述和图像路径
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    
    with open(encoded_captions_path, 'r') as f:
        encoded_captions = json.load(f)
    
    with open(image_paths_path, 'r') as f:
        image_paths = json.load(f)
    
    # 将索引转换回单词
    vocab_idx2word = {idx: word for word, idx in vocab.items()}
    
    # 选择要展示的图片ID，这里以第一个ID为例
    first_img_id = list(image_paths.keys())[0]
    content_img = Image.open(image_paths[first_img_id])
    
    # 展示图片和对应的描述
    plt.imshow(content_img)
    plt.axis('off')  # 不显示坐标轴
    plt.show()
    
    # 打印对应的文本描述，确保字典中的键是整数，直接使用整数索引
    caption = ' '.join([vocab_idx2word[word_idx] for word_idx in encoded_captions[first_img_id]])
    print(caption)
    ```

#### 4.2.3 模型定义

还没写...

#### 4.2.4 模型训练

还没写...

### 4.3 Transformer Encoder and Decoder

## 五、实验结果与分析

### 5.1 输出效果展示

### 5.2 评测指标统计

### 5.3 对比分析

可能包括不同模型的对比，误差分析，局限性分析和可能的改进方向。

## 六、总结

在进行实验的过程中，我们深刻体会到了许多关于计算机视觉和自然语言处理交叉领域的知识应用，关于经验和心得，可说的包括但不限于以下内容：

首先是模型选择，在选择模型结构时，我们意识到了模型的复杂性与性能之间的平衡。过于简单的模型可能无法捕捉到复杂的图像语境，而过于复杂的模型可能导致过拟合，而且可能会对硬件设备的要求比较高，综合考虑之下，我们选择了比较合适的几个模型。

其次是数据预处理部分，实验中，我们一开始没有对数据进行预处理，结果得到的效果不尽如人意，后来我们着手进行了数据的预处理，使得模型能够更好地理解输入数据，并提高了训练的效果。

紧接着是超参数的调优，通过多次实验，我们认识到超参数的选择和模型微调对实验结果的影响是巨大的。系统地调整学习率、Batch Size 和 Epoch 次数等超参数，结合模型微调，对于提高模型性能起到了关键作用。尤其是 Batch Size，这直接影响到了我们是否能够开始训练，在实验中，我们遇到了一个棘手的问题是内存不足：

![image](../doc/img/Out_of_Memory.png)

经过不断的查找解决方法，最终我们发现调整 Batch Size 和使用梯度积累的方法可以改善这种情况。

然后是评估指标的综合考虑，在评估模型性能时，我采用了多个评估指标，包括 METEOR、ROUGE-L 等。这帮助我更全面地了解了模型生成描述的质量。综合考虑不同指标的结果，有助于更全面地评估模型的性能。部分计算评估指标的代码如下：

```Python
# 计算METEOR分数
def calc_meteor(reference, hypothesis):
    hypothesis = word_tokenize(hypothesis)
    reference = word_tokenize(reference)
    return single_meteor_score(reference, hypothesis)

# 计算ROUGE-L分数
def calc_rouge_l(reference, hypothesis):
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference)
    return scores[0]['rouge-l']['f']
```

最后，通过这个实验，我们深感计算机视觉和自然语言处理的快速发展，也认识到了自己的知识不足之处。我们经过这次实验，对深度学习、图像处理和文本生成等领域有了更深入的理解，也激发了我们对未来深入学习和探索的兴趣。如果要用一句话来概括这整个神经网络与深度学习课程设计，我们会说这次实验是我们从理论到实践的一次重要尝试，通过不断调整和优化，我们逐渐提高了对这一复杂任务的理解，同时也加深了对深度学习技术的认识。这次实验不仅是对知识的巩固，也是对实际问题解决能力的锻炼，为我们未来的研究和工作奠定了坚实的基础，受益匪浅。

## 七、口头报告大纲

1. 任务描述
2. 功能和效果演示
3. 其他关键特征（创新性能力或表现、存在的问题或遇到的难题等）

