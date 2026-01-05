import torch
import torch.nn as nn


# SEBlock 保持不变，它对画质只有好处
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# # === 创新点 2: 深度可分离卷积 (DSC) ===
# class DSC_Conv(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(DSC_Conv, self).__init__()
#         # 深度卷积 (Depthwise): groups = in_ch
#         self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=True)
#         # 逐点卷积 (Pointwise): kernel_size = 1
#         self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         x = self.depthwise(x)
#         x = self.relu(x)
#         x = self.pointwise(x)
#         x = self.relu(x)
#         return x


class C_DCE_Net(nn.Module):
    def __init__(self, number_f=32):
        super(C_DCE_Net, self).__init__()

        # === 激进修改：全员标准卷积 ===
        # 放弃轻量化，追求极致画质

        # 第 1 层
        self.e_conv1 = nn.Conv2d(3, number_f, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        # 第 2 层 (原 DSC) -> 改标准卷积
        self.e_conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)

        # 第 3 层 (原 DSC) -> 改标准卷积
        self.e_conv3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)

        # Attention
        self.att1 = SEBlock(number_f)

        # 第 4 层 (原 DSC) -> 改标准卷积
        self.e_conv4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)

        # 第 5 层 (原 DSC) -> 改标准卷积
        # 注意输出通道是 number_f * 2
        self.e_conv5 = nn.Conv2d(number_f, number_f * 2, 3, 1, 1, bias=True)

        # Attention
        self.att2 = SEBlock(number_f * 2)

        # 第 6 层 (原 DSC) -> 改标准卷积
        self.e_conv6 = nn.Conv2d(number_f * 2, number_f * 2, 3, 1, 1, bias=True)

        # 第 7 层 (本来就是标准卷积)
        self.e_conv7 = nn.Conv2d(number_f * 2, 24, 3, 1, 1, bias=True)

    def forward(self, x):
        # 注意：标准卷积不带 ReLU，所以每一层后面都要手动加 self.relu

        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))  # 手动加 ReLU
        x3 = self.relu(self.e_conv3(x2))  # 手动加 ReLU

        x3 = self.att1(x3)

        x4 = self.relu(self.e_conv4(x3))  # 手动加 ReLU
        x5 = self.relu(self.e_conv5(x4))  # 手动加 ReLU

        x5 = self.att2(x5)

        x6 = self.relu(self.e_conv6(x5))  # 手动加 ReLU

        # 最后一层不需要 ReLU，因为它直接输出参数
        enhance_params = self.e_conv7(x6)

        x_r = torch.tanh(enhance_params)

        enhanced_image = x
        for i in range(8):
            r = x_r[:, i * 3: (i + 1) * 3, :, :]
            enhanced_image = enhanced_image + r * enhanced_image * (1 - enhanced_image)

        return enhanced_image, x_r