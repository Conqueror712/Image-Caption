import json
import torch
import os
from configurations import Config
from models import ARCTIC, get_optimizer, PackedCrossEntropyLoss, evaluate_cider
from datasets import create_dataloaders, ImageTextDataset


def main():
    # 加载配置
    config = Config()

    # 创建数据加载器
    train_loader, test_loader = create_dataloaders(config)

    # 加载词汇表文件
    with open('../data/output/vocab.json', 'r') as f:
        vocab = json.load(f)

    # 模型初始化
    model = ARCTIC(
        image_code_dim=config.image_code_dim,
        vocab=vocab,  # 传递词汇表字典
        word_dim=config.word_dim,
        attention_dim=config.attention_dim,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers
    ).to(config.device)

    # 优化器
    optimizer = get_optimizer(model, config)

    # 损失函数
    loss_fn = PackedCrossEntropyLoss().to(config.device)

    # 创建保存权重的文件夹路径
    weights_dir = os.path.join(config.output_folder, 'weights')
    os.makedirs(weights_dir, exist_ok=True)

    best_val_score = float('-inf')  # 初始化最佳验证得分

    # 开始训练
    for epoch in range(config.num_epochs):
        # 训练模型
        model.train()
        for i, (imgs, caps, caplens) in enumerate(train_loader):
            imgs, caps = imgs.to(config.device), caps.to(config.device)
            caplens = caplens.cpu().to(torch.int64)

            optimizer.zero_grad()
            outputs, alphas, _, _, _ = model(imgs, caps, caplens)

            # 确保目标序列长度与模型输出匹配
            targets = caps[:, 1:]  # 假设targets是captions去除第一个<start>标记后的部分
            # print(f"Outputs shape: {outputs.shape}")
            # print(f"Targets shape: {targets.shape}")
            # print(f"Caplens: {caplens}")
            loss = loss_fn(outputs, targets, caplens)
            loss.backward()
            optimizer.step()

            # 打印/记录损失信息
            if (i + 1) % 100 == 0:
                print('Epoch [{epoch + 1}/{config.num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        # 在每个epoch结束时使用测试集评估模型
        current_test_score = evaluate_cider(test_loader, model, config)
        print(f"Epoch {epoch}: Test score = {current_test_score}")

        # 如果当前得分比之前的最佳得分要好，则保存模型
        # if current_val_score > best_val_score:
        #     best_val_score = current_val_score
        #     best_model_path = os.path.join(weights_dir, f'best_model_epoch_{epoch}.pth')
        #     torch.save(model.state_dict(), best_model_path)
        #     print(f"Saved new best model to {best_model_path}")

    # 训练完成后的最终评估
    final_test_score = evaluate_cider(test_loader, model, config)
    print(f"Final test score = {final_test_score}")

    # 训练完成后保存模型
    final_model_path = os.path.join(weights_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model to {final_model_path}")


if __name__ == '__main__':
    main()
