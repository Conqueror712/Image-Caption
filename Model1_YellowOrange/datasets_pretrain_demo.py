import json
from PIL import Image
import matplotlib.pyplot as plt

vocab_path = '../data/output/vocab.json'
encoded_captions_path = '../data/output/encoded_captions_train.json'
image_paths_path = '../data/output/image_paths_train.json'

# 读取词典、编码后的描述和图像路径
with open(vocab_path, 'r') as f:
    vocab = json.load(f)

with open(encoded_captions_path, 'r') as f:
    encoded_captions = json.load(f)

with open(image_paths_path, 'r') as f:
    image_paths = json.load(f)

# 将索引转换回单词
vocab_idx2word = {idx: word for word, idx in vocab.items()}

# 选择要展示的图片ID，这里以第一个ID为例
first_img_id = list(image_paths.keys())[0]
content_img = Image.open(image_paths[first_img_id])

# 展示图片和对应的描述
plt.imshow(content_img)
plt.axis('off')  # 不显示坐标轴
plt.show()

# 打印对应的文本描述，确保字典中的键是整数，直接使用整数索引
caption = ' '.join([vocab_idx2word[word_idx] for word_idx in encoded_captions[first_img_id]])
# caption = ' '.join([vocab_idx2word[str(word_idx)] for word_idx in encoded_captions[first_img_id]])
print(caption)
