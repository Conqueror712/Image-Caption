import torch
from torch.nn.utils.rnn import pad_sequence

# 假设你有以下三个序列
sequences = [torch.tensor([1, 2, 3]), torch.tensor([4, 5]), torch.tensor([6])]

# 使用pad_sequence函数将这些序列填充到相同的长度
padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)

print(padded_sequences)
