""" Full assembly of the parts to form the complete network """

from unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear  # 决定上采样方式（双线性插值或转置卷积

        # 下采样
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
         # 上采样
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        # 转换为输出特征图
        self.outc = OutConv(64, n_classes)
        # 将输出转化为概率值
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return self.sigm(logits)

if __name__ == '__main__':
    print('code started')
    import torch
    
    model = UNet(n_channels = 3, n_classes=3)
    input1 = torch.rand(1,3,112, 112)  # 批量大小为 1、通道数为 3、高度和宽度为 112 的图像数据
    output1 = model(input1)
    print(f'output1 shape is {output1.shape}')
    #Keep number for classes=3 for RGB and apply sigmoid in the last layer