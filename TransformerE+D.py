import torch
from torch import nn
from torch.nn import Transformer
from torchvision import models, transforms
from PIL import Image
import os
import json
from nltk.translate.meteor_score import single_meteor_score
from rouge import Rouge
from cider.cider import Cider


def load_image(image_path, transform=None):
    image = Image.open(image_path)
    if transform is not None:
        image = transform(image)
    return image


def load_descriptions(description_path):
    with open(description_path, 'r') as f:
        descriptions = json.load(f)
    return descriptions


def create_model():
    # 使用预训练的ResNet模型来提取图像特征
    encoder = models.resnet50(pretrained=True)
    encoder = nn.Sequential(*list(encoder.children())[:-1])

    # 创建Transformer模型
    transformer = Transformer()

    return encoder, transformer


def train_model(encoder, transformer, images, descriptions):
    # TODO: 实现模型的训练过程
    pass


def evaluate_model(encoder, transformer, images, descriptions):
    # TODO: 实现模型的评估过程
    pass


def generate_description(encoder, transformer, image):
    # TODO: 实现描述的生成过程
    pass