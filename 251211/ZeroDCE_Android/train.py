import torch
import torch.optim as optim
import torch.utils.data as data
import os
from model import C_DCE_Net
from loss import L_color, L_spa, L_exp, L_TV
from data_loader import LowLightDataset


def train():
    # === 参数设置 ===
    # 自动检测有没有显卡
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    lr = 0.0001
    weight_decay = 0.0001
    batch_size = 8
    num_epochs = 100  # 训练轮数

    # 数据集路径：对应你创建的 dataset/LOLdataset
    dataset_path = "dataset/LOLdataset"

    # === 初始化 ===
    model = C_DCE_Net().to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 损失函数
    L_color_loss = L_color().to(device)
    L_spa_loss = L_spa().to(device)
    L_exp_loss = L_exp(16, 0.6).to(device)
    L_TV_loss = L_TV().to(device)

    # 加载数据
    train_dataset = LowLightDataset(dataset_path)
    if len(train_dataset) == 0:
        return

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    print("Start Training...")

    # === 训练循环 ===
    for epoch in range(num_epochs):
        for i, img_low in enumerate(train_loader):
            img_low = img_low.to(device)

            optimizer.zero_grad()

            # 前向传播 (返回: 增强图, 曲线参数)
            enhanced_image, A_maps = model(img_low)

            # 计算各项Loss
            loss_tv = 200 * L_TV_loss(A_maps)
            loss_spa = 1 * torch.mean(L_spa_loss(img_low, enhanced_image))
            loss_col = 5 * torch.mean(L_color_loss(enhanced_image))
            loss_exp = 10 * torch.mean(L_exp_loss(enhanced_image))

            loss = loss_tv + loss_spa + loss_col + loss_exp

            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}], Loss: {loss.item():.4f}")

        # 保存中间权重（可选）
        if (epoch + 1) % 50 == 0:
            if not os.path.exists('weights'): os.makedirs('weights')
            torch.save(model.state_dict(), f"weights/Epoch_{epoch + 1}.pth")

    # === 保存最终模型 ===
    if not os.path.exists('weights'): os.makedirs('weights')
    torch.save(model.state_dict(), "weights/ZeroDCE_final.pth")
    print("Training finished! Model saved to weights/ZeroDCE_final.pth")

if __name__ == '__main__':
    train()