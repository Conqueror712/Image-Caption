import torch
import os
import json
from torch import nn
from torch.nn import Transformer
from torchvision import models, transforms
from PIL import Image
from nltk.translate.meteor_score import single_meteor_score
from rouge import Rouge
from cider.cider import Cider


# 包含所有在描述中出现过的单词的列表，此处只是一个示例
vocabulary = ['<start>', 'a', 'cat', 'is', 'on', 'the', 'table', '<end>']


class ImageCaptionDataset(torch.utils.data.Dataset):
    """
    功能：加载和处理图像和描述
    """
    def __init__(self, image_dir, description_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.descriptions = load_descriptions(description_file)

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.descriptions[idx]['image_id'])
        image = load_image(image_path, self.transform)
        description = self.descriptions[idx]['caption']
        return image, description


def load_image(image_path, transform=None):
    image = Image.open(image_path)
    if transform is not None:
        image = transform(image)
    return image


def load_descriptions(description_path):
    with open(description_path, 'r') as f:
        descriptions = json.load(f)
    return descriptions


def compute_loss(predicted, target):
    """
    功能：计算交叉熵损失
    """
    loss = nn.CrossEntropyLoss()
    return loss(predicted, target)


def create_model():
    # 使用预训练的ResNet模型来提取图像特征
    encoder = models.resnet50(pretrained=True)
    encoder = nn.Sequential(*list(encoder.children())[:-1])

    # 创建Transformer模型
    transformer = Transformer()

    return encoder, transformer


def train_model(encoder, transformer, image_dir, description_file):
    """
    功能：训练模型
    """
    dataset = ImageCaptionDataset(image_dir, description_file, transform=transforms.ToTensor())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(transformer.parameters()))

    for epoch in range(10):  # 训练10个epoch
        for images, descriptions in dataloader:
            optimizer.zero_grad()

            features = encoder(images)
            output = transformer(features)

            loss = compute_loss(output, descriptions)
            loss.backward()

            optimizer.step()


def output_to_description(output, vocabulary):
    # 假设output的形状是(batch_size, sequence_length, vocabulary_size)
    # 其中每个元素是对应单词的概率

    # 选择概率最高的单词
    _, predicted_indices = torch.max(output, dim=2)

    # 将单词索引转换为单词
    descriptions = []
    for indices in predicted_indices:
        description = [vocabulary[index] for index in indices]
        descriptions.append(' '.join(description))

    return descriptions


def evaluate_model(encoder, transformer, image_dir, description_file):
    """
    功能：模型评估
    """
    dataset = ImageCaptionDataset(image_dir, description_file, transform=transforms.ToTensor())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    rouge = Rouge()
    cider = Cider()

    total_meteor_score = 0
    total_rouge_score = 0
    total_cider_score = 0

    with torch.no_grad():
        for images, descriptions in dataloader:
            features = encoder(images)
            output = transformer(features)

            predicted_descriptions = output_to_description(output)

            for predicted, target in zip(predicted_descriptions, descriptions):
                total_meteor_score += single_meteor_score(target, predicted)
                total_rouge_score += rouge.get_scores(target, predicted)[0]['rouge-l']['f']
                total_cider_score += cider.compute_score([target], [predicted])[0]

    average_meteor_score = total_meteor_score / len(dataset)
    average_rouge_score = total_rouge_score / len(dataset)
    average_cider_score = total_cider_score / len(dataset)

    return average_meteor_score, average_rouge_score, average_cider_score


def generate_description(encoder, transformer, image_path):
    """
    功能：生成描述
    """
    image = load_image(image_path, transform=transforms.ToTensor())
    image = image.unsqueeze(0)  # 添加一个批量维度

    with torch.no_grad():
        features = encoder(image)
        output = transformer(features)

    description = output_to_description(output)

    return description