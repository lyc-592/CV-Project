import torch
import torch.nn as nn


class C_DCE_Net(nn.Module):
    def __init__(self, number_f=32):
        super(C_DCE_Net, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        # 7层卷积网络，用于估计曲线参数
        self.e_conv1 = nn.Conv2d(3, number_f, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(number_f, number_f * 2, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(number_f * 2, number_f * 2, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(number_f * 2, 24, 3, 1, 1, bias=True)

    def forward(self, x):
        # 1. 神经网络估计参数
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(x4))
        x6 = self.relu(self.e_conv6(x5))
        enhance_params = self.e_conv7(x6)

        # 核心：生成曲线参数图 A，限制在 [-1, 1]
        x_r = torch.tanh(enhance_params)

        # 2. 应用增强公式 (Iteration)
        # 这个数学过程被嵌入在模型里，方便安卓端直接使用
        _, _, H, W = x.shape
        enhanced_image = x
        for i in range(8):
            r = x_r[:, i * 3: (i + 1) * 3, :, :]
            enhanced_image = enhanced_image + r * enhanced_image * (1 - enhanced_image)

        # 训练时返回：(增强图, 参数图)。参数图用于计算 Loss
        return enhanced_image, x_r