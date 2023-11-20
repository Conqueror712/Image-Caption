import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
import json
import string

# Setup
SEED = 1337
np.random.seed(SEED)
torch.manual_seed(SEED)

# Config
IMAGE_SIZE = (500, 500)
VOCAB_SIZE = 50
SEQ_LENGTH = 5
EMBED_DIM = 512
FF_DIM = 512
ENCODER_NUM_HEADS = 2
DECODER_NUM_HEADS = 2
BATCH_SIZE = 64
EPOCHS = 20


class CaptioningModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(CaptioningModel, self).__init__()
        self.cnn_model = models.vgg16(pretrained=True)
        self.cnn_model.classifier = nn.Sequential(*list(self.cnn_model.classifier.children())[:-1])
        self.cnn_model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(4096, EMBED_DIM)  # Add a fully connected layer
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = self.cnn_model(x)
        x = self.fc(x)  # Pass the output of CNN model through the fully connected layer
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Load the dataset
def load_captions_data(filename):
    with open(filename, 'r') as f:
        captions = json.load(f)
    image_paths = list(captions.keys())
    text_data = list(captions.values())
    return image_paths, text_data


# Load the dataset
captions_mapping, text_data = load_captions_data("captions.json")

# 获取分词器
tokenizer = get_tokenizer('basic_english')

# 构建词汇表
def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(text_data), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# 定义向量化函数
def vectorization(text):
    return [vocab[token] for token in tokenizer(text)]


# Split the dataset into training and validation sets
def train_val_split(caption_data, train_size=0.8, shuffle=True):
    size = len(caption_data)
    indices = np.arange(size)
    if shuffle:
        np.random.shuffle(indices)
    train_samples = int(size * train_size)
    train_data = dict((k, caption_data[k]) for k in indices[:train_samples])
    valid_data = dict((k, caption_data[k]) for k in indices[train_samples:])
    return train_data, valid_data

# Vectorizing the text data
def custom_standardization(input_string):
    lowercase = input_string.lower()
    return lowercase.translate(str.maketrans('', '', string.punctuation))

# Pipeline for training
def decode_and_resize(img_path):
    img = Image.open(img_path)
    img = img.resize(IMAGE_SIZE)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    img = transform(img)
    return img

def process_input(img_path, captions):
    img = decode_and_resize(img_path)
    return img, captions

def make_dataset(images, captions):
    img_dataset = [decode_and_resize(img) for img in images]
    cap_dataset = [vectorization(caption) for caption in captions]
    cap_dataset = pad_sequence(cap_dataset, batch_first=True, padding_value=vocab["<pad>"])
    dataset = list(zip(img_dataset, cap_dataset))
    return dataset


# Building the model
class TransformerEncoder(nn.Module):
    def __init__(self, num_heads, ff_dim):
        super(TransformerEncoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=EMBED_DIM, nhead=num_heads, dim_feedforward=ff_dim)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

    def forward(self, src):
        src = src.permute(1, 0, 2)
        output = self.transformer_encoder(src)
        return output

class TransformerDecoder(nn.Module):
    def __init__(self, num_heads, ff_dim):
        super(TransformerDecoder, self).__init__()
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=EMBED_DIM, nhead=num_heads, dim_feedforward=ff_dim)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=1)

    def forward(self, tgt, memory):
        tgt = tgt.permute(1, 0, 2)
        output = self.transformer_decoder(tgt, memory)
        return output

def create_captioning_model(encoder, decoder):
    cnn_model = models.vgg16(pretrained=True)
    cnn_model.classifier = nn.Sequential(*list(cnn_model.classifier.children())[:-1])
    cnn_model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    model = nn.Sequential(cnn_model, encoder, decoder)
    return model

# Training the model
def train_model(model, train_dataset, valid_dataset):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    for epoch in range(EPOCHS):
        model.train()
        for i, (inputs, labels) in enumerate(train_dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            labels = labels.long()
            loss = criterion(outputs.view(-1, VOCAB_SIZE), labels.view(-1))
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            total_loss = 0
            for i, (inputs, labels) in enumerate(valid_dataloader):
                outputs = model(inputs)
                labels = labels.long()
                loss = criterion(outputs.view(-1, VOCAB_SIZE), labels.view(-1))
                total_loss += loss.item()
            print(f"Validation loss: {total_loss / len(valid_dataloader)}")
    torch.save(model.state_dict(), "caption_model.pth")
            

# Main function
def main():
    # Split the dataset into training and validation sets
    train_data, valid_data = train_val_split(captions_mapping)

    # Vectorizing the text data
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(map(tokenizer, text_data), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    vectorization = lambda x: [vocab[token] for token in tokenizer(custom_standardization(x))]

    # Create the dataset
    train_dataset = make_dataset(list(train_data.keys()), list(train_data.values()))
    valid_dataset = make_dataset(list(valid_data.keys()), list(valid_data.values()))

    # Create the model
    encoder = TransformerEncoder(ENCODER_NUM_HEADS, FF_DIM)
    decoder = TransformerDecoder(DECODER_NUM_HEADS, FF_DIM)
    caption_model = create_captioning_model(encoder, decoder)

    # Train the model
    train_model(caption_model, train_dataset)

if __name__ == "__main__":
    main()
