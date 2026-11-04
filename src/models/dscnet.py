"""
Mo hinh DSCNet (Depthwise Separable Convolution Network) - Phien ban Tinh gon (Compact)
Su dung Conv 1x1 (Pointwise Convolution) va Bottleneck de giam toi da tham so mo hinh
Phu hop hoan hao voi quy mo du lieu thuc te (~6k-8k anh), ngan ngua Overfitting tuyet doi
Chi ~53k tham so (giam ~50 lan so voi ban goc 2.68M), toc do Real-time cuc nhanh
"""

import torch
import torch.nn as nn

class DSBlock(nn.Module):
    """
    Khoi Depthwise Separable Convolution co Skip Connection
    - 3x3 Depthwise Conv: hoc dac trung khong gian (spatial feature)
    - 1x1 Pointwise Conv: hoc dac trung ket hop giua cac kenh (cross-channel feature)
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(DSBlock, self).__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)

        # 1. Depthwise Convolution 3x3 (groups = in_channels)
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True)
        )

        # 2. Pointwise Convolution 1x1 (ep / chuyen doi so kenh)
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        out = self.pointwise(self.depthwise(x))
        if self.use_residual:
            return x + out
        return out

class DSCNet(nn.Module):
    def __init__(self, num_classes=7, dropout_rate=0.2):
        super(DSCNet, self).__init__()

        # Stem: Conv 3x3, 3 -> 16, stride 2
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True)
        )

        # 5 Groups voi 1x1 Pointwise Convolutions (11 DSBlocks tong cong)
        # Group 1: 16 -> 24 (s=1), 24 -> 32 (s=2)
        self.group1 = nn.Sequential(
            DSBlock(16, 24, stride=1),
            DSBlock(24, 32, stride=2)
        )

        # Group 2: 32 -> 32 (s=1), 32 -> 48 (s=2)
        self.group2 = nn.Sequential(
            DSBlock(32, 32, stride=1),
            DSBlock(32, 48, stride=2)
        )

        # Group 3: 48 -> 48 (s=1), 48 -> 64 (s=2)
        self.group3 = nn.Sequential(
            DSBlock(48, 48, stride=1),
            DSBlock(48, 64, stride=2)
        )

        # Group 4: 64 -> 64 (s=1) x 2, 64 -> 96 (s=2)
        self.group4 = nn.Sequential(
            DSBlock(64, 64, stride=1),
            DSBlock(64, 64, stride=1),
            DSBlock(64, 96, stride=2)
        )

        # Group 5: 96 -> 96 (s=1)
        self.group5 = nn.Sequential(
            DSBlock(96, 96, stride=1)
        )

        # Classification Head voi Conv 1x1: 96 -> 128 -> GAP -> Linear
        self.head_conv = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU6(inplace=True)
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(128, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = self.group4(x)
        x = self.group5(x)
        x = self.head_conv(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    net = DSCNet(num_classes=7)
    dummy = torch.randn(2, 3, 224, 224)
    out = net(dummy)
    print("DSCNet output shape:", out.shape)
    print("DSCNet trainable parameters:", f"{count_parameters(net):,}")
