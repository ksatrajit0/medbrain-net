import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(1, channels // reduction)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc(self.avg_pool(x))
        mx = self.fc(self.max_pool(x))
        return self.sigmoid(avg + mx)


class SpatialAttention(nn.Module):
    def __init__(self, kernel=7):
        super().__init__()

        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel,
            padding=kernel // 2,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)

        x = torch.cat([avg, mx], dim=1)
        x = self.conv(x)

        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),

            CBAM(out_ch)
        )

    def forward(self, x):
        return self.block(x)

def upsample(x, ref):
    return F.interpolate(
        x,
        size=ref.shape[2:],
        mode="bilinear",
        align_corners=True
    )

class UNetPlusPlus(nn.Module):

    def __init__(
        self,
        in_channels=4,
        out_channels=3,
        base_filters=32,
        deep_supervision=True
    ):
        super().__init__()

        f = base_filters
        self.deep_supervision = deep_supervision
        
        self.enc1 = ConvBlock(in_channels, f)
        self.enc2 = ConvBlock(f, f*2)
        self.enc3 = ConvBlock(f*2, f*4)
        self.enc4 = ConvBlock(f*4, f*8)

        self.center = ConvBlock(f*8, f*16)

        self.pool = nn.MaxPool2d(2)

        self.dec4_1 = ConvBlock(f*8 + f*16, f*8)

        self.dec3_1 = ConvBlock(f*4 + f*8, f*4)
        self.dec3_2 = ConvBlock(f*4*2 + f*8, f*4)

        self.dec2_1 = ConvBlock(f*2 + f*4, f*2)
        self.dec2_2 = ConvBlock(f*2*2 + f*4, f*2)
        self.dec2_3 = ConvBlock(f*2*3 + f*4, f*2)

        self.dec1_1 = ConvBlock(f + f*2, f)
        self.dec1_2 = ConvBlock(f*2 + f*2, f)
        self.dec1_3 = ConvBlock(f*3 + f*2, f)
        self.dec1_4 = ConvBlock(f*4 + f*2, f)
        
        self.out1 = nn.Conv2d(f, out_channels, 1)
        self.out2 = nn.Conv2d(f, out_channels, 1)
        self.out3 = nn.Conv2d(f, out_channels, 1)
        self.out4 = nn.Conv2d(f, out_channels, 1)

    def forward(self, x):
        
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))
        center = self.center(self.pool(x4))
        
        x4_1 = self.dec4_1(torch.cat([
            x4,
            upsample(center, x4)
        ], dim=1))
        
        x3_1 = self.dec3_1(torch.cat([
            x3,
            upsample(x4, x3)
        ], dim=1))

        x3_2 = self.dec3_2(torch.cat([
            x3,
            x3_1,
            upsample(x4_1, x3)
        ], dim=1))
        
        x2_1 = self.dec2_1(torch.cat([
            x2,
            upsample(x3, x2)
        ], dim=1))

        x2_2 = self.dec2_2(torch.cat([
            x2,
            x2_1,
            upsample(x3_1, x2)
        ], dim=1))

        x2_3 = self.dec2_3(torch.cat([
            x2,
            x2_1,
            x2_2,
            upsample(x3_2, x2)
        ], dim=1))
        
        x1_1 = self.dec1_1(torch.cat([
            x1,
            upsample(x2, x1)
        ], dim=1))

        x1_2 = self.dec1_2(torch.cat([
            x1,
            x1_1,
            upsample(x2_1, x1)
        ], dim=1))

        x1_3 = self.dec1_3(torch.cat([
            x1,
            x1_1,
            x1_2,
            upsample(x2_2, x1)
        ], dim=1))

        x1_4 = self.dec1_4(torch.cat([
            x1,
            x1_1,
            x1_2,
            x1_3,
            upsample(x2_3, x1)
        ], dim=1))

        if self.deep_supervision:
            return [
                self.out1(x1_1),
                self.out2(x1_2),
                self.out3(x1_3),
                self.out4(x1_4)
            ]

        return self.out4(x1_4)
