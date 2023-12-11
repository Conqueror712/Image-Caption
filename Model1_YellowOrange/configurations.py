import torch
class Config:
    # 数据路径
    data_path = '../data/'
    images_path = '../data/images/'
    train_captions_path = '../data/train_captions.json'
    test_captions_path = '../data/test_captions.json'
    output_folder = '../data/output/'  # 输出文件夹的路径，用于存储词汇表和处理后的数据

    # 模型参数
    embed_size = 256
    vocab_size = 10000  # 根据实际情况调整
    num_layers = 3  # 定义循环神经网络（RNN）或其变体（如 LSTM 或 GRU）中的层数。
    num_heads = 8
    dropout = 0.1
    hidden_size = 512

    # 数据处理参数
    min_word_count = 5  # 词汇表中词的最小出现次数
    max_len = 200  # 假设描述的最大长度为200个词

    # 训练参数
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 30
    workers = 4  # 工作线程数
    encoder_learning_rate = 1e-4  # 编码器的学习率
    decoder_learning_rate = 1e-3  # 解码器的学习率
    lr_update = 10  # 每10轮降低学习速率

    # 图像预处理参数
    image_size = 256  # 图像缩放大小
    crop_size = 224  # 图像裁剪大小

    # 其他配置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'