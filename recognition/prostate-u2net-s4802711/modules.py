"""
Components for the 2D U-Net architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class RSU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, mid_ch: int, L: int = 7):
        super(RSU, self).__init__()
        self.L = L

        self.rebnconvin = self._make_conv_block(in_ch, out_ch)

        # encoder stages
        self.rebnconv1 = self._make_conv_block(out_ch, mid_ch)
        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconvs = nn.ModuleList()
        for i in range(2, L):
            self.rebnconvs.append(self._make_conv_block(mid_ch, mid_ch))

        self.rebnconvm = self._make_conv_block(mid_ch, mid_ch)

        # decoder stages
        self.rebnconvd = nn.ModuleList()
        for i in range(2, L):
            self.rebnconvd.append(self._make_conv_block(mid_ch * 2, mid_ch))

        self.rebnconvout = self._make_conv_block(mid_ch * 2, out_ch)

    def _make_conv_block(self, in_ch: int, out_ch: int):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x_in = self.rebnconvin(x)
        
        # Encoder
        x = x_in
        conv_outs = [x]
        for i in range(1, self.L - 1):
            x = self.pool(x)
            if i == 1:
                x = self.rebnconv1(x)
            else:
                x = self.rebnconvs[i-1](x)

            conv_outs.append(x)
        
        # Bottleneck
        x = self.pool(x)
        x = self.rebnconvm(x)
        
        # Decoder
        for i in range(self.L - 2):
            idx = self.L - 2 - i
            x = F.interpolate(x, size=conv_outs[idx].size()[2:], mode='bilinear', align_corners=True)
            x = torch.cat((x, conv_outs[idx]), dim=1)
            x = self.rebnconvd[i](x)

        # Skip and Output
        x = F.interpolate(x, size=x_in.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, x_in), dim=1)

        # Final convolution
        x = self.rebnconvout(x)
        
        # Residual connection
        return x + x_in

        
class U2Net(nn.Module):
    def __init__(self, in_ch=1, out_ch=4):
        super(U2Net, self).__init__()

        # ENCODER - reduced channels and depth
        self.stage1 = RSU(in_ch, 16, 16, L=4)      # 1 -> 16 channels, L=4 instead of 7
        self.pool12 = nn.MaxPool2d(2)

        self.stage2 = RSU(16, 32, 32, L=4)         # 16 -> 32 channels
        self.pool23 = nn.MaxPool2d(2)

        self.stage3 = RSU(32, 64, 64, L=3)         # 32 -> 64 channels
        self.pool34 = nn.MaxPool2d(2)

        # BOTTLENECK - removed stage4, stage5, stage6
        self.stage4 = RSU(64, 128, 128, L=3)       # 64 -> 128 channels

        # DECODER - matching reduced encoder
        self.stage3d = RSU(128 + 64, 64, 64, L=3)  # 192 -> 64 channels
        self.stage2d = RSU(64 + 32, 32, 32, L=4)   # 96 -> 32 channels
        self.stage1d = RSU(32 + 16, 16, 16, L=4)   # 48 -> 16 channels

        # Output Layer
        self.outconv = nn.Conv2d(16, out_ch, 1)

    def forward(self, x):
        # Encoder
        x1 = self.stage1(x)      # 16 channels
        x = self.pool12(x1)

        x2 = self.stage2(x)      # 32 channels
        x = self.pool23(x2)

        x3 = self.stage3(x)      # 64 channels
        x = self.pool34(x3)

        x4 = self.stage4(x)      # 128 channels (bottleneck)

        # Decoder with skip connections
        x3d = F.interpolate(x4, size=x3.size()[2:], mode='bilinear', align_corners=True)
        x3d = torch.cat((x3d, x3), dim=1)  # 128 + 64 = 192
        x3d = self.stage3d(x3d)             # -> 64 channels

        x2d = F.interpolate(x3d, size=x2.size()[2:], mode='bilinear', align_corners=True)
        x2d = torch.cat((x2d, x2), dim=1)  # 64 + 32 = 96
        x2d = self.stage2d(x2d)             # -> 32 channels

        x1d = F.interpolate(x2d, size=x1.size()[2:], mode='bilinear', align_corners=True)
        x1d = torch.cat((x1d, x1), dim=1)  # 32 + 16 = 48
        x1d = self.stage1d(x1d)             # -> 16 channels

        # Output
        out = self.outconv(x1d)

        return out