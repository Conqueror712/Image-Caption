import os
import json
import torch
import nltk
from transformers import BertTokenizer
from PIL import Image
from models import caption
from datasets import coco
from models.alice import single_meteor_scr, rl_scr
from configuration import Config

nltk.download('punkt')
nltk.download('wordnet')

image_folder = "../data_new/train_images"

count = 1

config = Config()
model,_ = caption.build_model(config)
weights = torch.load("image_caption_model.pth")
model.load_state_dict(weights)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
start_token = tokenizer.convert_tokens_to_ids(tokenizer._cls_token)
end_token = tokenizer.convert_tokens_to_ids(tokenizer._sep_token)

def create_caption_and_mask(start_token, max_length):
    caption_template = torch.zeros((1, max_length), dtype=torch.long)
    mask_template = torch.ones((1, max_length), dtype=torch.bool)
    caption_template[:, 0] = start_token
    mask_template[:, 0] = False
    return caption_template, mask_template

caption, cap_mask = create_caption_and_mask(start_token, config.max_position_embeddings)

@torch.no_grad()
def evaluate(image):
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

# with open('../data_new/test_captions.json', 'r') as f:
#     captions = json.load(f)

result_dict = {}

for filename in os.listdir(image_folder):
    image_path = os.path.join(image_folder, filename)
    image = Image.open(image_path)
    image = coco.val_transform(image)
    image = image.unsqueeze(0)
    
    # reference_description = captions.get(filename, "No description found.")

    output = evaluate(image)
    result = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
    # print("Image Path       = ", image_path)
    # print("Image ID         = ", count)
    print("Image ID = ", count, "Predict Caption  = ", result.capitalize())
    # print("Reference Caption= ", reference_description.capitalize())
    # meteor_score = single_meteor_scr(reference_description, result)
    # rouge_l_score = rl_scr(reference_description, result)
    # print("-----------------------------")
    # print("|| METEOR  Score =", round(meteor_score, 4), " ||")
    # print("|| ROUGE-L Score =", round(rouge_l_score, 4), " ||")
    # print("-----------------------------")
    count += 1
    result_dict[image_path] = result

with open('../data_new/Model2_train_captions.json', 'w') as f:
    json.dump(result_dict, f)
