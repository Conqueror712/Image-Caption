import torch
from PIL import Image
from torchvision import transforms
from models import AttentionModel
from configurations import Config
import json

def load_model(model_path, vocab, config):
    model = AttentionModel(
        image_code_dim=config.image_code_dim,
        vocab=vocab,  # 传递词汇表字典
        word_dim=config.word_dim,
        attention_dim=config.attention_dim,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers
    )
    model.load_state_dict(torch.load(model_path))
    model = model.to(config.device)
    model.eval()  # 将模型设置为评估模式
    return model

def process_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # 添加一个批次维度
    return image_tensor

def predict_caption(model, image_tensor, vocab, config):
    # 生成束搜索描述
    predictions = model.generate_by_beamsearch(image_tensor.to(config.device), config.beam_k, config.max_len)
    # 将词索引转换回文字
    idx_to_word = {idx: word for word, idx in vocab.items()}
    caption_words = [idx_to_word.get(word, '<unk>') for word in predictions[0]]
    caption = ' '.join(caption_words)
    return caption

def main():
    # 载入配置和词汇表
    config = Config()
    with open('../data/output/vocab_caption_1.json', 'r') as f:
        vocab = json.load(f)

    # 加载模型
    model_path = '../data/output/weights/.pth'  # 使用正确的模型文件路径
    model = load_model(model_path, vocab, config)

    # 处理图片并生成描述
    image_path = '../data/images_1/MEN-Denim-id_00000080-01_7_additional.jpg'  # 测试图片路径
    image_tensor = process_image(image_path)
    caption = predict_caption(model, image_tensor, vocab, config)

    print("Generated Caption:", caption)

if __name__ == '__main__':
    main()


"""
model = ...             # 加载模型

images_folder = "..."   # 图片文件夹路径
captions_dict = {}      # 字典

count = 1               # 计数

for filename in os.listdir(images_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img_path = os.path.join(images_folder, filename)
        
        # Load the image
        raw_image = Image.open(img_path).convert('RGB')
    
        generated_caption = ... # 生成caption

        print(f"No{count}", generated_caption)
        count += 1

        # Store the caption in the dictionary
        captions_dict[img_path] = generated_caption

# Save the dictionary to captions.json
output_path = "..." # 保存路径
with open(output_path, 'w') as json_file:
    json.dump(captions_dict, json_file, indent=4)

print(f"Captions saved to {output_path}")
"""