import os
import json
import torch
import argparse
import nltk
from transformers import BertTokenizer
from PIL import Image
from models import caption
from datasets import coco
from models.alice import single_meteor_scr, rl_scr
from configuration import Config

nltk.download('punkt')
nltk.download('wordnet')

parser = argparse.ArgumentParser(description='Image Captioning')
parser.add_argument('--img', type=str, help='Image Path', required=True)
# parser.add_argument('--v', type=str, help='version', default='v3')
# parser.add_argument('--checkpoint', type=str, help='checkpoint path', default=None)
args = parser.parse_args()
image_path = args.path
# version = args.v
# checkpoint_path = args.checkpoint

config = Config()

# 加载模型
model = torch.hub.load('saahiluppal/catr', 'v3', pretrained=True)

# 保存模型权重
torch.save(model.state_dict(), "image_caption_model.pth")

# if version == 'v1':
#     model = torch.hub.load('saahiluppal/catr', 'v1', pretrained=True)
# elif version == 'v2':
#     model = torch.hub.load('saahiluppal/catr', 'v2', pretrained=True)
# elif version == 'v3':
#     model = torch.hub.load('saahiluppal/catr', 'v3', pretrained=True)
# else:
#     print("Checking for checkpoint.")
#     if checkpoint_path is None:
#       raise NotImplementedError('No model to chose from!')
#     else:
#       if not os.path.exists(checkpoint_path):
#         raise NotImplementedError('Give valid checkpoint path')
#       print("Found checkpoint! Loading!")
#       model,_ = caption.build_model(config)
#       print("Loading Checkpoint...")
#       checkpoint = torch.load(checkpoint_path, map_location='cpu')
#       model.load_state_dict(checkpoint['model'])
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

with open('../data/test_captions.json', 'r') as f:
    captions = json.load(f)

filename = os.path.basename(image_path)
reference_description = captions.get(filename, "No description found.")

output = evaluate()
result = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
print("=====================================================================")
print("Predict Caption   = ", result.capitalize())
print("Reference Caption = ", reference_description.capitalize())
meteor_score = single_meteor_scr(reference_description, result)
rouge_l_score = rl_scr(reference_description, result)
print("-----------------------------")
print("|| METEOR  Score =", round(meteor_score, 4), " ||")
print("|| ROUGE-L Score =", round(rouge_l_score, 4), " ||")
print("-----------------------------")