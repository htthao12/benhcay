"""
Mo hinh SE-ResNet18 - Phien ban Tinh gon (Compact SE-ResNet18 voi Conv 1x1 Bottleneck & SE Block)
Ket hop kien truc Bottleneck (Conv 1x1) voi co che chu y kenh Squeeze-and-Excitation (SE Block)
Giam tham so tu 11.27M xuong ~128k tham so (giam ~88 lan), toi uu tuyet doi cho du lieu vua va nho
Giup mang tap trung vao cac dac trung benh hoc quan trong nhat tren tung kenh mau
"""

import torch
import torch.nn as nn

class SEBlock(nn.Module):
    """
    Khoi Squeeze-and-Excitation (Hu et al., 2018) su dung Conv 1x1:
    - Squeeze: Global Average Pooling (B, C, H, W) -> (B, C, 1, 1)
    - Excitation: 1x1 Conv (C -> C/reduction) -> ReLU -> 1x1 Conv (C/reduction -> C) -> Sigmoid
    - Scale: Nhan trong so chu y vao dac trung dau vao
    """
    def __init__(self, channels, reduction=8):
        super(SEBlock, self).__init__()
        mid_channels = max(4, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        weight = self.fc(x)
        return x * weight

class SEBottleneckBlock(nn.Module):
    """
    Khoi Bottleneck tich hop SE Attention:
    1. Conv 1x1 (Squeeze channels)
    2. Conv 3x3 (Spatial feature)
    3. Conv 1x1 (Expand channels)
    4. SE Attention Block (Channel-wise weighting)
    5. Skip Connection + ReLU
    """
    expansion = 2

    def __init__(self, in_channels, base_channels, stride=1, downsample=None, reduction=8):
        super(SEBottleneckBlock, self).__init__()
        out_channels = base_channels * self.expansion

        # 1. Conv 1x1
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)

        # 2. Conv 3x3
        self.conv2 = nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(base_channels)

        # 3. Conv 1x1
        self.conv3 = nn.Conv2d(base_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # 4. SE Block
        self.se = SEBlock(out_channels, reduction=reduction)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        # Ap dung co che chu y kenh truoc khi cong shortcut
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class SEResNet18(nn.Module):
    def __init__(self, num_classes=7, dropout_rate=0.2, reduction=8):
        super(SEResNet18, self).__init__()
        self.in_channels = 16
        self.reduction = reduction

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 4 Stages voi SEBottleneckBlock
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
        self.fc = nn.Linear(48 * SEBottleneckBlock.expansion, num_classes)

        self._init_weights()

    def _make_stage(self, base_channels, blocks, stride=1):
        out_channels = base_channels * SEBottleneckBlock.expansion
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        layers = [SEBottleneckBlock(self.in_channels, base_channels, stride, downsample, self.reduction)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(SEBottleneckBlock(self.in_channels, base_channels, reduction=self.reduction))

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
    net = SEResNet18(num_classes=7)
    dummy = torch.randn(2, 3, 224, 224)
    out = net(dummy)
    print("SE-ResNet18 output shape:", out.shape)
    print("SE-ResNet18 trainable parameters:", f"{count_parameters(net):,}")
