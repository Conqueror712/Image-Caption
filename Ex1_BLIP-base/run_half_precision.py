import torch
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

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
print("===================================================================================")
print("更详细的描述：", processor.decode(out[0], skip_special_tokens=True))
print("===================================================================================")

# Unconditional image captioning
inputs = processor(raw_image, return_tensors="pt").to("cuda", torch.float16)

out = model.generate(**inputs)
print("===================================================================================")
print("更通用的描述：", processor.decode(out[0], skip_special_tokens=True))
print("===================================================================================")
