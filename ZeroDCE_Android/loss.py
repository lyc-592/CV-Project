import torch
import torch.nn as nn
import torch.nn.functional as F


class L_color(nn.Module):
    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x):
        b, c, h, w = x.shape
        # 计算三个通道的平均值
        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = mean_rgb[:, 0, :, :], mean_rgb[:, 1, :, :], mean_rgb[:, 2, :, :]
        # 强迫 R, G, B 三个通道的均值趋于一致（解决色偏，让光色更白/自然）
        d_rg = torch.pow(mr - mg, 2)
        d_rb = torch.pow(mr - mb, 2)
        d_gb = torch.pow(mg - mb, 2)
        k = torch.pow(torch.pow(d_rg, 2) + torch.pow(d_rb, 2) + torch.pow(d_gb, 2), 0.5)
        return k


class L_spa(nn.Module):
    def __init__(self):
        super(L_spa, self).__init__()
        # 卷积核用于计算梯度（边缘）
        kernel_left = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)

    def forward(self, org, enhance):
        b, c, h, w = org.shape
        # 为了加快计算并提取主要结构，先做平均池化
        org_mean = torch.mean(org, 1, keepdim=True)
        enhance_mean = torch.mean(enhance, 1, keepdim=True)

        org_pool = self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)

        # 计算原图和增强图在四个方向上的梯度差异
        # 这里的核心逻辑是：如果原图这里是平的，增强图也得是平的（抑制噪点）
        # 如果原图这里有边缘，增强图也要保留边缘
        D_org_letf = F.conv2d(org_pool, self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool, self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool, self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool, self.weight_down, padding=1)

        D_enhance_letf = F.conv2d(enhance_pool, self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool, self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool, self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool, self.weight_down, padding=1)

        D_left = torch.pow(D_org_letf - D_enhance_letf, 2)
        D_right = torch.pow(D_org_right - D_enhance_right, 2)
        D_up = torch.pow(D_org_up - D_enhance_up, 2)
        D_down = torch.pow(D_org_down - D_enhance_down, 2)

        return D_left + D_right + D_up + D_down


class L_exp(nn.Module):
    def __init__(self, patch_size, mean_val):
        super(L_exp, self).__init__()
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val

    def forward(self, x):
        b, c, h, w = x.shape
        x = torch.mean(x, 1, keepdim=True)
        mean = self.pool(x)
        # 强迫局部均值接近 mean_val (通常 0.5 或 0.6)
        d = torch.mean(torch.pow(mean - self.mean_val, 2))
        return d


class L_TV(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(L_TV, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        # 计算水平和垂直方向的平滑度
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size