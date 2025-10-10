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

        # ENCODER
        # Each RSU outputs out_ch channels (due to residual connection)
        self.stage1 = RSU(in_ch, 32, 16, L=7)      # 1 -> 32 channels
        self.pool12 = nn.MaxPool2d(2)

        self.stage2 = RSU(32, 32, 16, L=6)         # 32 -> 32 channels
        self.pool23 = nn.MaxPool2d(2)

        self.stage3 = RSU(32, 64, 32, L=5)         # 32 -> 64 channels
        self.pool34 = nn.MaxPool2d(2)

        self.stage4 = RSU(64, 128, 64, L=4)        # 64 -> 128 channels
        self.pool45 = nn.MaxPool2d(2)

        # BOTTLENECK
        self.stage5 = RSU(128, 256, 128, L=4)      # 128 -> 256 channels
        self.pool56 = nn.MaxPool2d(2)

        self.stage6 = RSU(256, 512, 256, L=4)      # 256 -> 512 channels

        # DECODER
        # Concatenate upsampled decoder output with skip connection
        self.stage5d = RSU(512 + 256, 256, 128, L=4)  # 768 -> 256 channels
        self.stage4d = RSU(256 + 128, 128, 64, L=4)   # 384 -> 128 channels
        self.stage3d = RSU(128 + 64, 64, 32, L=5)     # 192 -> 64 channels
        self.stage2d = RSU(64 + 32, 32, 16, L=6)      # 96 -> 32 channels
        self.stage1d = RSU(32 + 32, 32, 16, L=7)      # 64 -> 32 channels

        # Output Layer
        self.outconv = nn.Conv2d(32, out_ch, 1)

    def forward(self, x):
        # Encoder
        x1 = self.stage1(x)      # 32 channels
        x = self.pool12(x1)

        x2 = self.stage2(x)      # 32 channels
        x = self.pool23(x2)

        x3 = self.stage3(x)      # 64 channels
        x = self.pool34(x3)

        x4 = self.stage4(x)      # 128 channels
        x = self.pool45(x4)

        x5 = self.stage5(x)      # 256 channels
        x = self.pool56(x5)

        x6 = self.stage6(x)      # 512 channels

        # Decoder with skip connections
        x5d = F.interpolate(x6, size=x5.size()[2:], mode='bilinear', align_corners=True)
        x5d = torch.cat((x5d, x5), dim=1)  # 512 + 256 = 768
        x5d = self.stage5d(x5d)             # -> 256 channels

        x4d = F.interpolate(x5d, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x4d = torch.cat((x4d, x4), dim=1)  # 256 + 128 = 384
        x4d = self.stage4d(x4d)             # -> 128 channels

        x3d = F.interpolate(x4d, size=x3.size()[2:], mode='bilinear', align_corners=True)
        x3d = torch.cat((x3d, x3), dim=1)  # 128 + 64 = 192
        x3d = self.stage3d(x3d)             # -> 64 channels

        x2d = F.interpolate(x3d, size=x2.size()[2:], mode='bilinear', align_corners=True)
        x2d = torch.cat((x2d, x2), dim=1)  # 64 + 32 = 96
        x2d = self.stage2d(x2d)             # -> 32 channels

        x1d = F.interpolate(x2d, size=x1.size()[2:], mode='bilinear', align_corners=True)
        x1d = torch.cat((x1d, x1), dim=1)  # 32 + 32 = 64
        x1d = self.stage1d(x1d)             # -> 32 channels

        # Output
        out = self.outconv(x1d)

        return out