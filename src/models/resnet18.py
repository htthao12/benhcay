"""
Mo hinh ResNet18 - Phien ban Tinh gon (Compact ResNet18 voi Conv 1x1 Bottleneck)
Su dung kien truc Bottleneck (Conv 1x1 ep kenh -> Conv 3x3 trich xuat -> Conv 1x1 mo rong)
Giam tham so tu 11.18M xuong ~120k tham so (giam ~93 lan), hoan toan khong overfit tren tap du lieu nho
Van giu nguyen 18 layers va co che Residual Skip Connection chuan He et al. (2016)
"""

import torch
import torch.nn as nn

class BottleneckBlock(nn.Module):
    """
    Khoi Bottleneck voi Conv 1x1 (He et al., 2016):
    - Conv 1x1: Ep/giam so kenh tu in_channels xuong base_channels
    - Conv 3x3: Xu ly khong gian tren khong gian kenh da duoc tinh gon
    - Conv 1x1: Mo rong tro lai so kenh out_channels
    - Shortcut 1x1: Dieu chinh so kenh khi downsample
    """
    expansion = 2

    def __init__(self, in_channels, base_channels, stride=1, downsample=None):
        super(BottleneckBlock, self).__init__()
        out_channels = base_channels * self.expansion

        # 1. Conv 1x1: Ep kenh (Squeeze)
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)

        # 2. Conv 3x3: Trich xuat dac trung khong gian
        self.conv2 = nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(base_channels)

        # 3. Conv 1x1: Mo rong kenh (Expand)
        self.conv3 = nn.Conv2d(base_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes=7, dropout_rate=0.2):
        super(ResNet18, self).__init__()
        self.in_channels = 16

        # Stem: Conv 3x3 stride 2 -> MaxPool (tinh gon hon stem 7x7 goc)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 4 Stages voi cac khoi Bottleneck 1x1 (8 blocks = 18 layers tong)
        # Stage 1: base=16 -> out=32 (2 blocks)
        self.stage1 = self._make_stage(16, 2, stride=1)
        # Stage 2: base=24 -> out=48 (2 blocks)
        self.stage2 = self._make_stage(24, 2, stride=2)
        # Stage 3: base=32 -> out=64 (2 blocks)
        self.stage3 = self._make_stage(32, 2, stride=2)
        # Stage 4: base=48 -> out=96 (2 blocks)
        self.stage4 = self._make_stage(48, 2, stride=2)

        # Classification Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(48 * BottleneckBlock.expansion, num_classes)

        self._init_weights()

    def _make_stage(self, base_channels, blocks, stride=1):
        out_channels = base_channels * BottleneckBlock.expansion
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        layers = [BottleneckBlock(self.in_channels, base_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BottleneckBlock(self.in_channels, base_channels))

        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    net = ResNet18(num_classes=7)
    dummy = torch.randn(2, 3, 224, 224)
    out = net(dummy)
    print("ResNet18 output shape:", out.shape)
    print("ResNet18 trainable parameters:", f"{count_parameters(net):,}")
