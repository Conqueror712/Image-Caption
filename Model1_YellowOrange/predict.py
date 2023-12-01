import torch
from models import ImageCaptioningModel
from datasets import FashionDataset
from configuration import Config

def predict(image_path, model_path):
    # 加载配置和模型
    config = Config()
    model = ImageCaptioningModel(config.embed_size, config.vocab_size, config.num_layers, config.num_heads, config.dropout)
    model.load_state_dict(torch.load(model_path))
    model.to(config.device)
    model.eval()

    # 加载和处理图像
    dataset = FashionDataset(config.images_path, None)
    image = dataset.load_image(image_path)
    image = image.to(config.device)

    # 生成描述
    with torch.no_grad():
        caption = model.generate(image)
    return caption
