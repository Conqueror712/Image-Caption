import os
import json
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", torch_dtype=torch.float16).to("cuda")

# Path to the folder containing images
images_folder = "../Ex_data/test_images"

# Create a dictionary to store image captions
captions_dict = {}

count = 1

# Loop through each file in the images folder
for filename in os.listdir(images_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Add other supported image formats if necessary
        img_path = os.path.join(images_folder, filename)
        
        # Load the image
        raw_image = Image.open(img_path).convert('RGB')

        # Your text for captioning
        text = "The background content in the picture is"
        
        # Process the image and generate caption
        inputs = processor(raw_image, text, return_tensors="pt").to("cuda", torch.float16)
        out = model.generate(**inputs)
        generated_caption = processor.decode(out[0], skip_special_tokens=True)

        print(f"No{count}", generated_caption)
        count += 1

        # Store the caption in the dictionary
        captions_dict[img_path] = generated_caption

# Save the dictionary to captions.json
output_path = "background_test_captions.json"
with open(output_path, 'w') as json_file:
    json.dump(captions_dict, json_file, indent=4)

print(f"Captions saved to {output_path}")
