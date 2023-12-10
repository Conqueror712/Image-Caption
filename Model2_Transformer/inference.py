import os
import json
import torch
import argparse
import nltk
from transformers import BertTokenizer
from PIL import Image
from models import caption
from datasets import coco, utils
from configuration import Config
from nltk.translate.meteor_score import single_meteor_score
from nltk.tokenize import word_tokenize
from rouge import Rouge

nltk.download('punkt')
nltk.download('wordnet')

# 读取命令行参数
parser = argparse.ArgumentParser(description='Image Captioning')
parser.add_argument('--path', type=str, help='path to image', required=True)
args = parser.parse_args()
image_path = args.path

# 加载模型
config = Config()
model, _ = caption.build_model(config)
model.load_state_dict(torch.load('Model2.pth'))
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 预处理图片
start_token = tokenizer.convert_tokens_to_ids(tokenizer._cls_token)
end_token = tokenizer.convert_tokens_to_ids(tokenizer._sep_token)
image = Image.open(image_path)
image = coco.val_transform(image)
image = image.unsqueeze(0)

# 创建 caption 和 mask
def create_caption_and_mask(start_token, max_length):
    caption_template = torch.zeros((1, max_length), dtype=torch.long)
    mask_template = torch.ones((1, max_length), dtype=torch.bool)

    caption_template[:, 0] = start_token
    mask_template[:, 0] = False

    return caption_template, mask_template

caption, cap_mask = create_caption_and_mask(start_token, config.max_position_embeddings)

# 生成 caption
@torch.no_grad()
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

# 计算 METEOR 分数
def calc_meteor(reference, hypothesis):
    hypothesis = word_tokenize(hypothesis)
    reference = word_tokenize(reference)
    return single_meteor_score(reference, hypothesis)

# 计算 ROUGE-L 分数
def calc_rouge_l(reference, hypothesis):
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference)
    return scores[0]['rouge-l']['f']

with open('../data/test_captions.json', 'r') as f:
    captions = json.load(f)

filename = os.path.basename(image_path)
reference_description = captions.get(filename, "No description found.")

output = evaluate()
result = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
print("=====================================================================")
print("Predict Caption   = ", result.capitalize())
print("Reference Caption = ", reference_description.capitalize())

meteor_score = calc_meteor(reference_description, result)
rouge_l_score = calc_rouge_l(reference_description, result)
print("-----------------------------")
print("|| METEOR  Score =", round(meteor_score, 4), " ||")
print("|| ROUGE-L Score =", round(rouge_l_score, 4), " ||")
print("-----------------------------")