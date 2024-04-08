import torch
import torch.nn as nn
import torch.nn.functional as F


class SKFusion(nn.Module):
    def __init__(self, dim, height=2, reduction=4):
        super(SKFusion, self).__init__()

        self.height = height
        d = max(int(dim / reduction), 8)

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.mlp = nn.Sequential(
            nn.Conv3d(dim, d, 1, bias=False),
            nn.LeakyReLU(),
            nn.Conv3d(d, dim * height, 1, bias=False),
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        B, C, D, H, W = in_feats[0].shape

        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, D, H, W)

        feats_sum = torch.sum(in_feats, dim=1)
        attn = self.mlp(self.avg_pool(feats_sum))
        attn = self.softmax(attn.view(B, self.height, C, 1, 1, 1))

        out = torch.sum(in_feats * attn, dim=1)
        return out


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
        # self.double_conv = nn.Sequential(
        #     nn.Conv3d(in_channels, out_channels, 3, 1, 1),
        #     nn.InstanceNorm3d(out_channels),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv3d(out_channels, out_channels, 3, 1, 1),
        #     nn.InstanceNorm3d(out_channels),
        #     nn.LeakyReLU(inplace=True),
        # )
        self.residual = in_channels == out_channels

    def forward(self, x):
        return self.double_conv(x) + x if self.residual else self.double_conv(x)


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
    def __init__(self, in_channels, n_classes, n_channels, deep_supervision=True):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.n_channels = n_channels

        self.conv = DoubleConv(in_channels, n_channels)
        self.enc1 = Down(n_channels, 2 * n_channels)
        self.enc2 = Down(2 * n_channels, 4 * n_channels)
        self.enc3 = Down(4 * n_channels, 8 * n_channels)
        self.enc4 = Down(8 * n_channels, 8 * n_channels)

        self.dec1 = Up(16 * n_channels, 4 * n_channels)
        self.dec2 = Up(8 * n_channels, 2 * n_channels)
        self.dec3 = Up(4 * n_channels, n_channels)
        self.dec4 = Up(2 * n_channels, n_channels)
        self.deep_supervision = deep_supervision
        self.out = nn.ModuleList(
            [
                Out(4 * n_channels, n_classes),
                Out(2 * n_channels, n_classes),
                Out(n_channels, n_classes),
                mschead(n_channels, n_classes),
            ]
        )

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        x5 = self.enc4(x4)

        mask1 = self.dec1(x5, x4)
        mask2 = self.dec2(mask1, x3)
        mask3 = self.dec3(mask2, x2)
        mask4 = self.dec4(mask3, x1)
        masks = [mask1, mask2, mask3, mask4]

        if self.deep_supervision:
            return [m(mask) for m, mask in zip(self.out, masks)][::-1]
        else:
            return self.out[-1](mask4)


if __name__ == "__main__":
    model = MyNet(1, 2, 32, False)
    print(model.state_dict().keys())
    # x = torch.randn((2, 1, 128, 128, 128))
    # y = model(x)
    # for _ in y:
    #     print(_.shape)