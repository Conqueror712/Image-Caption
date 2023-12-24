import torch
import torch.nn as nn
from pycocoevalcap.cider.cider import Cider
import numpy as np
from configurations import Config
from torchvision.models import resnet101, ResNet101_Weights
from torch.nn.utils.rnn import pack_padded_sequence
import torch.optim as optim
import json
import torchvision


# 图像编码器
class ImageEncoder(nn.Module):
    def __init__(self, finetuned=True):
        super(ImageEncoder, self).__init__()
        model = torchvision.models.resnet101(weights=ResNet101_Weights.DEFAULT)
        # ResNet-101网格表示提取器
        self.grid_rep_extractor = nn.Sequential(*(list(model.children())[:-2]))
        for param in self.grid_rep_extractor.parameters():
            param.requires_grad = finetuned

    def forward(self, images):
        out = self.grid_rep_extractor(images)
        return out

# # 引入自注意机制后的图像编码器
# class SelfAttention(nn.Module):
#     def __init__(self, num_channels, num_heads=8, dropout=0.1):
#         super(SelfAttention, self).__init__()
#         self.num_heads = num_heads
#         self.attention = nn.MultiheadAttention(num_channels, num_heads, dropout)
#
#     def forward(self, x):
#         # 保存原始形状
#         orig_shape = x.shape
#         # 打印输入形状
#         # print("Input shape:", x.shape)
#         # 转换为(sequence_length, batch_size, num_channels)格式
#         x = x.flatten(2).permute(2, 0, 1)
#         attention_output, _ = self.attention(x, x, x)
#         # 还原形状，确保与原始输入形状匹配
#         attention_output = attention_output.permute(1, 2, 0)# 打印最终输出形状
#         # print("Final output shape:", attention_output.shape)
#         return attention_output.view(orig_shape)
#
#
# class ImageEncoder(nn.Module):
#     def __init__(self, finetuned=True, num_heads=8, dropout=0.1):
#         super(ImageEncoder, self).__init__()
#         # 使用ResNet101作为基础模型
#         model = resnet101(weights=ResNet101_Weights.DEFAULT)
#         self.grid_rep_extractor = nn.Sequential(*(list(model.children())[:-2]))
#         # 设置参数是否可训练
#         for param in self.grid_rep_extractor.parameters():
#             param.requires_grad = finetuned
#
#         # 自注意力层
#         self.self_attention = SelfAttention(model.fc.in_features, num_heads, dropout)
#
#     def forward(self, images):
#         features = self.grid_rep_extractor(images)
#         features = self.self_attention(features)
#         return features


# 解码器的注意力机制
class AdditiveAttention(nn.Module):
    def  __init__(self, query_dim, key_dim, attn_dim):
        super(AdditiveAttention, self).__init__()
        self.attn_w_1_q = nn.Linear(query_dim, attn_dim)
        self.attn_w_1_k = nn.Linear(key_dim, attn_dim)
        self.attn_w_2 = nn.Linear(attn_dim, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, query, key_value):
        queries = self.attn_w_1_q(query).unsqueeze(1)
        keys = self.attn_w_1_k(key_value)
        attn = self.attn_w_2(self.tanh(queries+keys)).squeeze(2)
        attn = self.softmax(attn)
        output = torch.bmm(attn.unsqueeze(1), key_value).squeeze(1)
        return output, attn


# 文本解码器
class AttentionDecoder(nn.Module):
    def __init__(self, image_code_dim, vocab_size, word_dim, attention_dim, hidden_size, num_layers, dropout=0.5):
        super(AttentionDecoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, word_dim)
        self.attention = AdditiveAttention(hidden_size, image_code_dim, attention_dim)
        self.init_state = nn.Linear(image_code_dim, num_layers * hidden_size)
        self.rnn = nn.GRU(word_dim + image_code_dim, hidden_size, num_layers)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, image_code, captions, cap_lens):
        batch_size, image_code_dim = image_code.size(0), image_code.size(1)
        image_code = image_code.permute(0, 2, 3, 1)
        image_code = image_code.view(batch_size, -1, image_code_dim)
        sorted_cap_lens, sorted_cap_indices = torch.sort(cap_lens, 0, True)
        captions = captions[sorted_cap_indices]
        image_code = image_code[sorted_cap_indices]
        hidden_state = self.init_state(image_code.mean(axis=1))
        hidden_state = hidden_state.view(
            batch_size,
            self.rnn.num_layers,
            self.rnn.hidden_size).permute(1, 0, 2)
        return image_code, captions, sorted_cap_lens, sorted_cap_indices, hidden_state

    def forward_step(self, image_code, curr_cap_embed, hidden_state):
        context, alpha = self.attention(hidden_state[-1], image_code)
        x = torch.cat((context, curr_cap_embed), dim=-1).unsqueeze(0)
        out, hidden_state = self.rnn(x, hidden_state)
        preds = self.fc(self.dropout(out.squeeze(0)))
        return preds, alpha, hidden_state

    def forward(self, image_code, captions, cap_lens):
        image_code, captions, sorted_cap_lens, sorted_cap_indices, hidden_state \
            = self.init_hidden_state(image_code, captions, cap_lens)
        batch_size = image_code.size(0)
        lengths = sorted_cap_lens.cpu().numpy() - 1
        max_cap_len = max(cap_lens)
        predictions = torch.zeros(batch_size, max_cap_len, self.fc.out_features).to(captions.device)
        alphas = torch.zeros(batch_size, max_cap_len, image_code.shape[1]).to(captions.device)
        cap_embeds = self.embed(captions)
        # Teacher-Forcing模式
        for step in range(lengths[0]):
            real_batch_size = np.where(lengths > step)[0].shape[0]
            preds, alpha, hidden_state = self.forward_step(
                image_code[:real_batch_size],
                cap_embeds[:real_batch_size, step, :],
                hidden_state[:, :real_batch_size, :].contiguous())
            predictions[:real_batch_size, step, :] = preds
            alphas[:real_batch_size, step, :] = alpha
            max_cap_len = max(cap_lens)
            padded_predictions = torch.zeros(batch_size, max_cap_len, self.fc.out_features).to(predictions.device)
            for i in range(batch_size):
                actual_length = cap_lens[i]
                padded_predictions[i, :actual_length, :] = predictions[i, :actual_length, :]

        return padded_predictions, alphas, captions, lengths, sorted_cap_indices


class ARCTIC(nn.Module):
    def __init__(self, image_code_dim, vocab, word_dim, attention_dim, hidden_size, num_layers):
        super(ARCTIC, self).__init__()
        self.vocab = vocab
        self.encoder = ImageEncoder()
        self.decoder = AttentionDecoder(image_code_dim, len(vocab), word_dim, attention_dim, hidden_size, num_layers)

    def forward(self, images, captions, cap_lens):
        image_code = self.encoder(images)
        output = self.decoder(image_code, captions, cap_lens)
        return output

    def generate_by_beamsearch(self, images, beam_k, max_len):
        vocab_size = len(self.vocab)
        image_codes = self.encoder(images)
        texts = []
        device = images.device
        for image_code in image_codes:
            image_code = image_code.unsqueeze(0).repeat(beam_k, 1, 1, 1)
            cur_sents = torch.full((beam_k, 1), self.vocab['<start>'], dtype=torch.long).to(device)
            cur_sent_embed = self.decoder.embed(cur_sents)[:, 0, :]
            sent_lens = torch.LongTensor([1] * beam_k).to(device)
            image_code, cur_sent_embed, _, _, hidden_state = \
                self.decoder.init_hidden_state(image_code, cur_sent_embed, sent_lens)
            end_sents = []
            end_probs = []
            probs = torch.zeros(beam_k, 1).to(device)
            k = beam_k
            while True:
                preds, _, hidden_state = self.decoder.forward_step(image_code[:k], cur_sent_embed,
                                                                   hidden_state.contiguous())
                preds = nn.functional.log_softmax(preds, dim=1)
                probs = probs.repeat(1, preds.size(1)) + preds
                if cur_sents.size(1) == 1:
                    values, indices = probs[0].topk(k, 0, True, True)
                else:
                    values, indices = probs.view(-1).topk(k, 0, True, True)
                sent_indices = torch.div(indices, vocab_size, rounding_mode='trunc')
                word_indices = indices % vocab_size
                cur_sents = torch.cat([cur_sents[sent_indices], word_indices.unsqueeze(1)], dim=1)
                end_indices = [idx for idx, word in enumerate(word_indices) if word == self.vocab['<end>']]
                if len(end_indices) > 0:
                    end_probs.extend(values[end_indices])
                    end_sents.extend(cur_sents[end_indices].tolist())
                    k -= len(end_indices)
                    if k == 0:
                        break
                cur_indices = [idx for idx, word in enumerate(word_indices)
                               if word != self.vocab['<end>']]
                if len(cur_indices) > 0:
                    cur_sent_indices = sent_indices[cur_indices]
                    cur_word_indices = word_indices[cur_indices]
                    cur_sents = cur_sents[cur_indices]
                    probs = values[cur_indices].view(-1, 1)
                    hidden_state = hidden_state[:, cur_sent_indices, :]
                    cur_sent_embed = self.decoder.embed(
                        cur_word_indices.view(-1, 1))[:, 0, :]
                if cur_sents.size(1) >= max_len:
                    break
            if len(end_sents) == 0:
                gen_sent = cur_sents[0].tolist()
            else:
                gen_sent = end_sents[end_probs.index(max(end_probs))]
            texts.append(gen_sent)
        return texts


# 损失函数
class PackedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(PackedCrossEntropyLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, predictions, targets, lengths):
        packed_predictions = pack_padded_sequence(predictions, lengths, batch_first=True, enforce_sorted=False)[0]
        packed_targets = pack_padded_sequence(targets, lengths, batch_first=True, enforce_sorted=False)[0]

        # 计算损失，忽略填充的部分
        loss = self.loss_fn(packed_predictions, packed_targets)
        return loss


def get_optimizer(model, config):
    encoder_params = filter(lambda p: p.requires_grad, model.encoder.parameters())
    decoder_params = filter(lambda p: p.requires_grad, model.decoder.parameters())
    optimizer = optim.Adam([
        {"params": encoder_params, "lr": config.encoder_learning_rate},
        {"params": decoder_params, "lr": config.decoder_learning_rate}
    ])

    return optimizer

def adjust_learning_rate(optimizer, epoch, config):
    for param_group in optimizer.param_groups:
        if param_group['name'] == 'encoder':
            param_group['lr'] = config.encoder_learning_rate * (0.1 ** (epoch // config.lr_update))
        else:
            param_group['lr'] = config.decoder_learning_rate * (0.1 ** (epoch // config.lr_update))


# CIDEr-D 评估
def filter_useless_words(sent, filterd_words):
    return [w for w in sent if w not in filterd_words]


def evaluate_cider(data_loader, model, config):
    model.eval()
    # 存储候选文本和参考文本
    cands = {}
    refs = {}
    filterd_words = {model.vocab['<start>'], model.vocab['<end>'], model.vocab['<pad>']}
    device = next(model.parameters()).device

    # 加载词汇表并创建反向词汇表
    with open('../output_副本/vocab.json', 'r') as f:
        vocab = json.load(f)
    idx_to_word = {idx: word for word, idx in vocab.items()}

    for i, (imgs, caps, caplens) in enumerate(data_loader):
        imgs = imgs.to(device)
        preds = model.generate_by_beamsearch(imgs, config.beam_k, config.max_len)
        for j in range(imgs.size(0)):
            img_id = str(i * config.batch_size + j)
            cand_words = [idx_to_word.get(word, '<unk>') for word in preds[j]]
            cand = ' '.join(filter_useless_words(cand_words, filterd_words))
            cands[img_id] = [cand]
            ref_words = [idx_to_word.get(word.item(), '<unk>') for word in caps[j]]
            refs[img_id] = [' '.join(filter_useless_words(ref_words, filterd_words))]  # 参考描述

    # 计算CIDEr-D得分
    cider_evaluator = Cider()
    score, _ = cider_evaluator.compute_score(refs, cands)

    model.train()
    return score