import torch
import torch.optim as optim
import torch.utils.data as data
import os
from model import C_DCE_Net
from loss import L_color, L_spa, L_exp, L_TV
from data_loader import LowLightDataset


def train():
    # === 参数设置 ===
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    lr = 0.0001
    weight_decay = 0.0001

    # 修改: 适配 512x512 分辨率，降低 BatchSize 防止爆显存
    batch_size = 4
    num_epochs = 100

    dataset_path = "dataset/LOLdataset"

    # === 初始化 ===
    model = C_DCE_Net().to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Loss 初始化
    L_color_loss = L_color().to(device)
    L_spa_loss = L_spa().to(device)
    L_exp_loss = L_exp(16, 0.5).to(device)
    L_TV_loss = L_TV().to(device)

    train_dataset = LowLightDataset(dataset_path)
    if len(train_dataset) == 0:
        return

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    print("Start Training (Optimized for CLARITY and SHARPNESS)...")

    # === 训练循环 ===
    for epoch in range(num_epochs):
        for i, img_low in enumerate(train_loader):
            img_low = img_low.to(device)

            optimizer.zero_grad()

            enhanced_image, A_maps = model(img_low)

            # === 核心修改: 调整 Loss 权重 ===

            # 1. TV Loss (平滑度): 1600 -> 200
            # 作用: 哪怕会有少量噪点，也要大幅减少“涂抹感”，保留纹理细节
            loss_tv = 200 * L_TV_loss(A_maps)

            # 2. Spa Loss (空间一致性): 5 -> 20
            # 作用: 强力锁死边缘，让物体轮廓像原图一样清晰锐利
            loss_spa = 20 * torch.mean(L_spa_loss(img_low, enhanced_image))

            # 3. Color Loss: 5 (不变)
            loss_col = 5 * torch.mean(L_color_loss(enhanced_image))

            # 4. Exp Loss: 10 (不变)
            loss_exp = 10 * torch.mean(L_exp_loss(enhanced_image))

            loss = loss_tv + loss_spa + loss_col + loss_exp

            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}], Loss: {loss.item():.4f}")

        if (epoch + 1) % 50 == 0:
            if not os.path.exists('weights'): os.makedirs('weights')
            torch.save(model.state_dict(), f"weights/Epoch_{epoch + 1}.pth")

    if not os.path.exists('weights'): os.makedirs('weights')
    torch.save(model.state_dict(), "weights/ZeroDCE_final.pth")
    print("Training finished! Model saved to weights/ZeroDCE_final.pth")


if __name__ == '__main__':
    train()