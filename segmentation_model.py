import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(out))

class CBAM(nn.Module):
    def __init__(self, in_channels):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        out = x * self.ca(x)
        return out * self.sa(out)

def conv_block(in_channels, out_channels, activation=nn.LeakyReLU(0.1, True), dropout_p=0.2):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        activation,
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        activation,
        CBAM(out_channels),
        nn.Dropout2d(p=dropout_p)
    )

class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, deep_supervision=True, n_filters=32):
        super().__init__()
        self.deep_supervision = deep_supervision
        act = nn.LeakyReLU(0.1, True)
        
        # Encoder
        self.enc1 = conv_block(in_channels, n_filters, act, 0.1)
        self.enc2 = conv_block(n_filters, 2*n_filters, act, 0.1)
        self.enc3 = conv_block(2*n_filters, 4*n_filters, act, 0.2)
        self.enc4 = conv_block(4*n_filters, 8*n_filters, act, 0.2)
        self.center = conv_block(8*n_filters, 16*n_filters, act, 0.3)

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Nested Decoder Blocks (L1, L2, L3, L4 paths)
        self.dec4_1 = conv_block(24*n_filters, 8*n_filters, act, 0.2) 
        self.dec3_1 = conv_block(12*n_filters, 4*n_filters, act, 0.2)
        self.dec3_2 = conv_block(12*n_filters, 4*n_filters, act, 0.2)
        self.dec2_1 = conv_block(6*n_filters, 2*n_filters, act, 0.1)
        self.dec2_2 = conv_block(6*n_filters, 2*n_filters, act, 0.1)
        self.dec2_3 = conv_block(6*n_filters, 2*n_filters, act, 0.1)
        self.dec1_1 = conv_block(3*n_filters, n_filters, act, 0.1)
        self.dec1_2 = conv_block(3*n_filters, n_filters, act, 0.1)
        self.dec1_3 = conv_block(3*n_filters, n_filters, act, 0.1)
        self.dec1_4 = conv_block(3*n_filters, n_filters, act, 0.1)

        self.outputs = nn.ModuleList([nn.Conv2d(n_filters, out_channels, 1) for _ in range(4)])

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))
        ctr = self.center(self.pool(x4))

        # Nested Decoding
        x4_1 = self.dec4_1(torch.cat([x4, self.up(ctr)], 1))
        
        x3_1 = self.dec3_1(torch.cat([x3, self.up(x4)], 1))
        x3_2 = self.dec3_2(torch.cat([x3, self.up(x4_1)], 1))
        
        x2_1 = self.dec2_1(torch.cat([x2, self.up(x3)], 1))
        x2_2 = self.dec2_2(torch.cat([x2, self.up(x3_1)], 1))
        x2_3 = self.dec2_3(torch.cat([x2, self.up(x3_2)], 1))
        
        x1_1 = self.dec1_1(torch.cat([x1, self.up(x2)], 1))
        x1_2 = self.dec1_2(torch.cat([x1, self.up(x2_1)], 1))
        x1_3 = self.dec1_3(torch.cat([x1, self.up(x2_2)], 1))
        x1_4 = self.dec1_4(torch.cat([x1, self.up(x2_3)], 1))

        if self.deep_supervision:
            return [self.outputs[0](x1_1), self.outputs[1](x1_2), 
                    self.outputs[2](x1_3), self.outputs[3](x1_4)]
        return self.outputs[3](x1_4)