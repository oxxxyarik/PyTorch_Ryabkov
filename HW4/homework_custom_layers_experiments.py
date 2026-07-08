import torch
import torch.nn as nn

# Кастомный слой Channel Attention (SE-Block)
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze & Excite
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# Кастомный Bottleneck Residual Block
class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, bottleneck_channels):
        super().__init__()
        # 1x1 Сжатие
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        # 3x3 Свертка
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        # 1x1 Расширение обратно
        self.conv3 = nn.Conv2d(bottleneck_channels, in_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(in_channels)
        
        self.attention = ChannelAttention(in_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        # Применяем Attention к выходу перед сложением с residual
        out = self.attention(out)
        out += residual
        return self.relu(out)

if __name__ == '__main__':
    # Тестовый прогон проверки кастомных тензорных размерностей
    print("Testing custom layers geometry consistency...")
    test_tensor = torch.randn(2, 64, 32, 32) # [Batch, Channels, Height, Width]
    
    block = BottleneckBlock(in_channels=64, bottleneck_channels=16)
    output = block(test_tensor)
    
    print(f"Input shape:  {test_tensor.shape}")
    print(f"Output shape: {output.shape}")
    if test_tensor.shape == output.shape:
        print("Success: BottleneckBlock forward pass preserves dimensions.")