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


class MSCHead(nn.Module):
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
        )
        self.residual = in_channels == out_channels
        self.leaky_relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        return (
            self.leaky_relu(self.double_conv(x) + x)
            if self.residual
            else self.leaky_relu(self.double_conv(x))
        )


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 2, 2),
            DoubleConv(in_channels, out_channels),
            DoubleConv(out_channels, out_channels),
        )

    def forward(self, x):
        return self.encoder(x)


class Up(nn.Module):
    def __init__(
        self,
        low_channels,
        high_channels,
        out_channels,
        fusion_mode="add",
    ):
        super().__init__()

        self.fusion_mode = fusion_mode
        self.up = nn.ConvTranspose3d(
            low_channels, high_channels, kernel_size=2, stride=2
        )
        self.conv = (
            DoubleConv(2 * high_channels, out_channels)
            if fusion_mode == "cat"
            else DoubleConv(high_channels, out_channels)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if self.fusion_mode == "cat":
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x2 + x1
        return self.conv(x)


class MyNet(nn.Module):
    """
    MyNet

    Args:
        in_channels (int): 输入通道数。
        n_classes (int): 类别数。
        depth (int, optional): 编码器和解码器的深度。默认为 4。
        head_channels (int, optional): 头部卷积层的通道数。默认为 16。
        encoder_channels (list of int, optional): 编码器每一层的通道数。
            列表长度必须等于 `depth`。默认为 [16, 32, 64, 128, 256]。

    Raises:
        AssertionError: 如果输入参数不满足条件。
    """

    def __init__(
        self,
        in_channels,
        n_classes,
        depth=4,
        encoder_channels=[16, 32, 64, 128, 256],
        head_channels=16,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.depth = depth
        self.head_channels = head_channels
        assert len(encoder_channels) == depth + 1, "len(encoder_channels) != depth + 1"

        self.conv = DoubleConv(in_channels, encoder_channels[0])
        self.encoders = nn.ModuleList()  # 使用 ModuleList 存储编码器层
        self.decoders = nn.ModuleList()  # 使用 ModuleList 存储解码器层

        # 创建编码器层
        for i in range(self.depth):
            self.encoders.append(Down(encoder_channels[i], encoder_channels[i + 1]))

        # 创建解码器层
        for i in range(self.depth):
            self.decoders.append(
                Up(
                    encoder_channels[self.depth - i],
                    encoder_channels[self.depth - i - 1],
                    encoder_channels[self.depth - i - 1],
                )
            )

        self.params_list = [
            encoder_channels[0] * head_channels,
            head_channels * n_classes,
            head_channels,
            n_classes,
        ]

        self.GAP = nn.AdaptiveAvgPool3d(1)
        self.controller = nn.Conv3d(
            encoder_channels[0] * 2**self.depth + 2, sum(self.params_list), 1
        )

    def encoding_task(self, task_id):
        N = task_id.shape[0]
        task_encoding = torch.zeros(size=(N, 2))
        for i in range(N):
            task_encoding[i, task_id[i]] = 1
        return task_encoding.cuda()

    def forward(self, x, task_id):
        encoder_features = [self.conv(x)]  # 存储编码器输出

        # 编码过程
        for encoder in self.encoders:
            encoder_features.append(encoder(encoder_features[-1]))

        # 解码过程
        x_dec = encoder_features[-1]
        decoder_features = []  # 用于存储解码器特征
        for i, decoder in enumerate(self.decoders):
            x_dec = decoder(x_dec, encoder_features[-(i + 2)])
            decoder_features.append(x_dec)  # 保存解码器特征

        task_embeddings = self.encoding_task(task_id)
        task_embeddings = task_embeddings.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x_feat = self.GAP(encoder_features[-1])
        x_cond = torch.cat([x_feat, task_embeddings], dim=1)
        params = self.controller(x_cond)
        params = params.squeeze(-1).squeeze(-1).squeeze(-1)
        params_split = torch.split_with_sizes(params, self.params_list, dim=1)
        N, _, D, H, W = x_dec.shape
        head_feat = x_dec.view(1, -1, D, H, W)
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

    model = MyNet(1, 2, 3, [16, 32, 64, 128]).cuda()
    # print(model.state_dict().keys())
    x = torch.randn((2, 1, 192, 192, 192)).cuda()
    y = model(x, np.array([0, 1]))
    for _ in y:
        print(_.shape)
