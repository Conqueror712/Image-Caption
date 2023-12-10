import os
import json
import shutil

# 读取json文件并转换为字典
with open('../data/test_captions.json', 'r') as f:
    captions = json.load(f)

# 指定源目录和目标目录
source_directory = 'data/images'
train_directory = 'data/train_images'
test_directory = 'data/test_images'

# 确保目标目录存在
os.makedirs(train_directory, exist_ok=True)
os.makedirs(test_directory, exist_ok=True)

# 遍历字典中的所有文件名
for filename in captions.keys():
    # 构造源文件和目标文件的完整路径
    source_file = os.path.join(source_directory, filename)
    train_file = os.path.join(train_directory, filename)
    test_file = os.path.join(test_directory, filename)
    # 复制文件
    shutil.copy(source_file, train_file)
    shutil.copy(source_file, test_file)