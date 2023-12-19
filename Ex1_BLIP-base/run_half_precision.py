import os
import json
import requests
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from nltk.tokenize import word_tokenize
from nltk.translate.meteor_score import single_meteor_score
from rouge import Rouge

def print_line():
    print("============================================================================================")

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

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", torch_dtype=torch.float16).to("cuda")

# img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' # 图片URL版本
# raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')         # 图片URL版本

img_local_url = "../data/test_images/MEN-Denim-id_00000089-17_4_full.jpg"               # 本地图片版本
raw_image = Image.open(img_local_url).convert('RGB')                                    # 本地图片版本

# Conditional image captioning
text = "a photography of"
inputs = processor(raw_image, text, return_tensors="pt").to("cuda", torch.float16)

out = model.generate(**inputs)
generated_caption = processor.decode(out[0], skip_special_tokens=True)
print_line()
print("更详细的描述：", generated_caption)
print_line()

# Unconditional image captioning
inputs = processor(raw_image, return_tensors="pt").to("cuda", torch.float16)

out = model.generate(**inputs)
generated_caption_unconditional = processor.decode(out[0], skip_special_tokens=True)
print_line()
print("更通用的描述：", generated_caption_unconditional)
print_line()

# 加入评估指标计算
with open('../data/test_captions.json', 'r') as f:
    captions = json.load(f)

filename = os.path.basename(img_local_url)
reference_description = captions.get(filename, "No description found.")

print_line()
print("Predict Caption   = ", generated_caption.capitalize())
print("Reference Caption = ", reference_description.capitalize())

meteor_score = calc_meteor(reference_description, generated_caption)
rouge_l_score = calc_rouge_l(reference_description, generated_caption)
print_line()
print("METEOR  Score =", round(meteor_score, 4))
print("ROUGE-L Score =", round(rouge_l_score, 4))
print_line()
