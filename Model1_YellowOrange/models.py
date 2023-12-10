import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from transformers import BertModel, BertConfig

# 图像编码器
class EncoderCNNwithAttention(nn.Module):
    def __init__(self, embed_size, num_layers, num_heads, dropout=0.1):
        super(EncoderCNNwithAttention, self).__init__()
        # 加载预训练的ResNet
        resnet = models.resnet50(pretrained=True)
        # 移除最后的全连接层，此时网络的输出是一个高维特征向量，它代表了图像的内容
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.transformer_enc_layer = TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, dropout=dropout)
        self.transformer_enc = TransformerEncoder(self.transformer_enc_layer, num_layers=num_layers)

    def forward(self, images):
        features = self.resnet(images)
        features = self.flatten(features)
        features = self.linear(features)
        features = features.unsqueeze(1)
        features = self.transformer_enc(features)
        return features


# 文本解码器 - 使用Transformer结构
class DecoderTransformer(nn.Module):
    def __init__(self, embed_size, vocab_size, num_layers, num_heads, dropout=0.1):
        super(DecoderTransformer, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)  # 词嵌入层
        self.positional_encoding = PositionalEncoding(embed_size, dropout)  # 位置编码
        transformer_layer = TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, dropout=dropout)
        self.transformer_decoder = TransformerEncoder(transformer_layer, num_layers)  # Transformer解码器
        self.fc_out = nn.Linear(embed_size, vocab_size)  # 最后输出层

    def forward(self, features, captions):
        embeddings = self.embed(captions)  # 对文本进行嵌入
        embeddings = self.positional_encoding(embeddings)  # 添加位置编码
        transformer_out = self.transformer_decoder(embeddings + features)  # 将图像特征和文本嵌入相加，送入Transformer
        outputs = self.fc_out(transformer_out)  # 得到每个词的输出分布
        return outputs

# 位置编码层，给解码器输入的嵌入向量添加位置信息
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# 图像描述生成模型
class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_size, vocab_size, num_layers, num_heads, dropout=0.1):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = EncoderCNNwithAttention(embed_size, num_layers, num_heads, dropout)  # 使用前面定义的图像编码器
        self.decoder = DecoderTransformer(embed_size, vocab_size, num_layers, num_heads, dropout)  # 使用上面定义的文本解码器

    def forward(self, images, captions):
        features = self.encoder(images)  # 得到图像的特征
        outputs = self.decoder(features, captions)  # 使用图像特征和文本生成描述
        return outputs



# # 初始化模型用到的全局变量，此处需要提供词汇表大小
# embed_size = 256  # 嵌入层大小
# vocab_size = len(vocab)  # 词汇表的大小，vocab需要事先定义
# num_layers = 3  # Transformer层的数量
# num_heads = 8  # 多头注意力的头数
# dropout = 0.1  # Dropout率
#
# model = ImageCaptioningModel(embed_size, vocab_size, num_layers, num_heads, dropout)
# model.to(device)  # 将模型传输到GPU设备上
#
# # 示例用法
# images = torch.randn(32, 3, 224, 224).to(device)  # 假设有一批32张224x224的图像
# # captions = torch.randint(0, vocab_size, (32, 20)).to(device)  # 假设每个图像有20个词的描述
# output = model(images, captions)  # 得到模型输出

