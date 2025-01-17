import torch
import torch.nn as nn
from thop import profile, clever_format

# 定义 U-Net 的双卷积块
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
# 定义 U-Net 模型
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 收缩路径
        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 中间层
        self.middle = DoubleConv(512, 1024)

        # 扩展路径
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv4 = DoubleConv(128, 64)

        # 输出层
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # 收缩路径
        x1 = self.down1(x)
        x2 = self.pool(x1)
        x3 = self.down2(x2)
        x4 = self.pool(x3)
        x5 = self.down3(x4)
        x6 = self.pool(x5)
        x7 = self.down4(x6)
        x8 = self.pool(x7)

        # 中间层
        x9 = self.middle(x8)

        # 扩展路径
        x = self.up1(x9)
        x = torch.cat([x, x7], dim=1)
        x = self.up_conv1(x)
        x = self.up2(x)
        x = torch.cat([x, x5], dim=1)
        x = self.up_conv2(x)
        x = self.up3(x)
        x = torch.cat([x, x3], dim=1)
        x = self.up_conv3(x)
        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up_conv4(x)

        # 输出层
        x = self.out_conv(x)
        return x
# 创建 U-Net 模型实例
model = UNet(in_channels=3, out_channels=1)


# 创建输入张量
input = torch.randn(1, 3, 224, 224)

# 使用 thop 库计算 FLOPs 和参数数量
macs, params = profile(model, inputs=(input, ))

# 计算 FLOPs
flops = 2 * macs  # 一个 MAC 操作等价于两次 FLOPs

# 使用 clever_format 函数格式化输出
flops, params = clever_format([flops, params], "%.3f")

# 打印结果
print("Params:", params)
print("GFLOPs:", flops)