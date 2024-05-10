from __future__ import annotations
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, repeat, einsum
from typing import Union
# from mamba_ssm import Mamba
from torch.cuda.amp import autocast


class MambaLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,  # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,  # Local convolution width
            expand=2,  # Block expansion factor
        )

    @autocast(enabled=False)
    def forward(self, x):
        b, c, d, h, w = x.shape
        x = x.permute(0, 2, 3, 4, 1).contiguous()  # b d h w c
        x = x.view(b, -1, c)  # 调整维度以适配后续层
        x = self.norm(x)  # 应用LayerNorm
        x = self.mamba(x)
        x = x.view(b, d, h, w, c)
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # b c d h w
        return x


class DoubleConv(nn.Module):
    """(Conv3D -> IN -> LeakyReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 1, 1, 0),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, 1, 1, groups=8),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 1, 1, 0),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
        )
        self.res_mamba = MambaLayer(d_model=out_channels)
        self.residual = in_channels == out_channels

    def forward(self, x):
        # 应用双卷积
        residual = x
        x = self.double_conv(x)
        if self.residual:
            x = x + residual  # 使用残差连接

        x = self.res_mamba(x)  # 应用Mamba层

        return x


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool3d(2, 2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.encoder(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, trilinear=False):
        super().__init__()

        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(
                in_channels // 2, in_channels // 2, kernel_size=2, stride=2
            )

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = F.pad(
            x1,
            [
                diffX // 2,
                diffX - diffX // 2,
                diffY // 2,
                diffY - diffY // 2,
                diffZ // 2,
                diffZ - diffZ // 2,
            ],
        )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class mschead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.conv3 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv3d(in_channels, in_channels, kernel_size=5, padding=2)
        self.conv7 = nn.Conv3d(in_channels, in_channels, kernel_size=7, padding=3)
        self.sk = SKFusion(in_channels, height=4)
        self.conv = nn.Conv3d(in_channels * 5, out_channels, kernel_size=7, padding=3)

    def forward(self, x):
        x1 = self.conv1(x)
        x3 = self.conv3(x1)
        x5 = self.conv5(x1)
        x7 = self.conv7(x1)
        x_sk = self.sk([x1, x3, x5, x7])
        x = self.conv(torch.cat([x1, x3, x5, x7, x_sk], dim=1))
        return x


class MyNet(nn.Module):
    def __init__(self, in_channels, n_classes, n_channels, head_channels=16):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.head_channels = head_channels

        self.conv = DoubleConv(in_channels, n_channels)
        self.enc1 = Down(n_channels, 2 * n_channels)
        self.enc2 = Down(2 * n_channels, 4 * n_channels)
        self.enc3 = Down(4 * n_channels, 8 * n_channels)
        self.enc4 = Down(8 * n_channels, 8 * n_channels)

        self.dec1 = Up(16 * n_channels, 4 * n_channels)
        self.dec2 = Up(8 * n_channels, 2 * n_channels)
        self.dec3 = Up(4 * n_channels, n_channels)
        self.dec4 = Up(2 * n_channels, n_channels)

        self.params_list = [
            n_channels * head_channels,
            head_channels * n_classes,
            head_channels,
            n_classes,
        ]

        self.GAP = nn.AdaptiveAvgPool3d(1)
        self.controller = nn.Conv3d(n_channels * 8 + 2, sum(self.params_list), 1)

    def encoding_task(self, task_id):
        N = task_id.shape[0]
        task_encoding = torch.zeros(size=(N, 2))
        for i in range(N):
            task_encoding[i, task_id[i]] = 1
        return task_encoding.cuda()

    def forward(self, x, task_id):
        x1 = self.conv(x)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        x5 = self.enc4(x4)

        mask1 = self.dec1(x5, x4)
        mask2 = self.dec2(mask1, x3)
        mask3 = self.dec3(mask2, x2)
        mask4 = self.dec4(mask3, x1)

        task_embeddings = self.encoding_task(task_id)
        task_embeddings = task_embeddings.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x_feat = self.GAP(x5)
        x_cond = torch.cat([x_feat, task_embeddings], dim=1)
        params = self.controller(x_cond)
        params = params.squeeze(-1).squeeze(-1).squeeze(-1)
        params_split = torch.split_with_sizes(params, self.params_list, dim=1)
        N, _, D, H, W = mask4.shape
        head_feat = mask4.view(1, -1, D, H, W)
        head_feat = F.leaky_relu(
            F.conv3d(
                head_feat,
                params_split[0].reshape(N * self.head_channels, -1, 1, 1, 1),
                bias=params_split[2].reshape(N * self.head_channels),
                stride=1,
                padding=0,
                groups=N,
            )
        )
        logits = F.conv3d(
            head_feat,
            params_split[1].reshape(self.n_classes * N, -1, 1, 1, 1),
            bias=params_split[3].reshape(self.n_classes * N),
            stride=1,
            padding=0,
            groups=N,
        )
        logits = logits.reshape(N, -1, D, H, W)
        return logits


if __name__ == "__main__":
    import numpy as np

    model = MyNet(1, 2, 32, 16).cuda()
    # print(model.state_dict().keys())
    x = torch.randn((2, 1, 48, 224, 128)).cuda()
    y = model(x, np.array([0, 1]))
    for _ in y:
        print(_.shape)
