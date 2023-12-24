"""
必须要指出的是，图像描述的任务，这是一个序列生成任务，而不是一个强化学习任务。
在这种情况下，使用强化学习可能并不是最好的选择，因为定义出合适的奖励函数可能会非常困难。
"""
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import BertTokenizer
from PIL import Image
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
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# 定义奖励函数
def reward_function(predictions, targets):
    # 这只是一个示例，你需要根据你的任务定义合适的奖励函数
    return (predictions == targets).float()

# 定义策略梯度更新函数
def policy_gradient_update(model, images, captions, optimizer):
    outputs = model(images, captions['input_ids'], captions['attention_mask'])
    rewards = reward_function(outputs.logits.argmax(-1), captions['input_ids'])
    action_probs = outputs.logits.softmax(-1)
    picked_action_probs = action_probs.gather(-1, captions['input_ids'].unsqueeze(-1)).squeeze(-1)
    loss = (-torch.log(picked_action_probs) * rewards).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss

# 训练循环
num_epochs = 10
for epoch in range(num_epochs):
    for images, captions in train_dataloader:
        images = images.to(device)  # 将图像数据移动到指定的设备上
        captions = tokenizer(captions, return_tensors='pt', padding=True, truncation=True)
        captions = {key: val.to(device) for key, val in captions.items()}  # 将caption数据移动到指定的设备上

        loss = policy_gradient_update(model, images, captions, optimizer)

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# 保存模型
torch.save(model.state_dict(), 'Model2.pth')