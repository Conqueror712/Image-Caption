import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import BertTokenizer
from PIL import Image
from models import caption
from configuration import Config

# 数据集类
class MyDataset(Dataset):
    def __init__(self, json_file, img_dir, transform=None):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.img_dir = img_dir
        self.transform = transform
        self.filenames = list(self.data.keys())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        caption = self.data[filename]
        image = Image.open(f"{self.img_dir}/{filename}")
        if self.transform:
            image = self.transform(image)
        return image, caption

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型
config = Config()
model = torch.hub.load('saahiluppal/catr', 'v3', pretrained=True)
model = model.to(device)  # 将模型移动到指定的设备上
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 准备数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = MyDataset('../data/train_captions.json', '../data/train_images', transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# 训练循环
num_epochs = 10
accumulation_steps = 4  # 梯度累积步数
optimizer.zero_grad()  # 在开始梯度累积之前，先清零梯度

for epoch in range(num_epochs):
    for i, (images, captions) in enumerate(train_dataloader):
        images = images.to(device)
        captions = tokenizer(captions, return_tensors='pt', padding=True, truncation=True)
        captions = {key: val.to(device) for key, val in captions.items()}

        outputs = model(images, captions['input_ids'], captions['attention_mask'])
        loss = criterion(outputs.logits.view(-1, outputs.logits.size(-1)), captions['input_ids'].view(-1))

        loss = loss / accumulation_steps  # Normalize the loss because it is averaged
        loss.backward()

        if (i+1) % accumulation_steps == 0:  # Wait for several backward steps
            optimizer.step()  # Now we can do an optimizer step
            optimizer.zero_grad()  # Reset gradients tensors

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# 保存模型
torch.save(model.state_dict(), 'Model2.pth')