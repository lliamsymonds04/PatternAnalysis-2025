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
        self.rebconvs = nn.ModuleList()
        for i in range(2, L):
            self.rebconvs.append(self._make_conv_block(mid_ch, mid_ch))

        self.rebconvm = self._make_conv_block(mid_ch, mid_ch)

        # decoder stages
        self.rebconvd = nn.ModuleList()
        for i in range(2, L):
            self.rebconvd.append(self._make_conv_block(mid_ch * 2, mid_ch))

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
        x = self.rebnconvout(x)
        
        # Residual connection
        return x + x_in

        
class U2Net(nn.Module):
    def __init__(self, in_ch=1, out_ch=4):
        super(U2Net, self).__init__()

        # ENCODER
        self.stage1 = RSU(in_ch, 32, 64, L=7)
        self.pool12 = nn.MaxPool2d(2)

        self.stage2 = RSU(64, 32, 128, L=6)
        self.pool23 = nn.MaxPool2d(2)

        self.stage3 = RSU(128, 64, 256, L=5)
        self.pool34 = nn.MaxPool2d(2)

        self.stage4 = RSU(256, 128, 512, L=4)
        self.pool45 = nn.MaxPool2d(2)

        # BOTTLENECK
        self.stage5 = RSU(512, 256, 512, L=4)
        self.pool56 = nn.MaxPool2d(2)

        self.stage6 = RSU(512, 256, 512, L=4)

        # DECODER
        self.stage5d = RSU(1024, 256, 512, L=4) # 512 from stage6 + 512 from stage5
        self.stage4d = RSU(1024, 128, 256, L=5) # 512 from stage5d + 512 from stage4
        self.stage3d = RSU(512, 64, 128, L=6)   # 256 from stage4d + 256 from stage3
        self.stage2d = RSU(256, 32, 64, L=7)    # 128 from stage3d + 128 from stage2
        self.stage1d = RSU(128, 16, 64, L=7)    # 64 from stage2d + 64 from stage1

        # Output Layer
        self.outconv = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        # Encoder
        x1 = self.stage1(x)
        x = self.pool12(x1)

        x2 = self.stage2(x)
        x = self.pool23(x2)

        x3 = self.stage3(x)
        x = self.pool34(x3)

        x4 = self.stage4(x)
        x = self.pool45(x4)

        x5 = self.stage5(x)
        x = self.pool56(x5)

        x6 = self.stage6(x)

        # Decoder
        x5d = F.interpolate(x6, size=x5.size()[2:], mode='bilinear', align_corners=True)
        x5d = torch.cat((x5d, x5), dim=1)
        x5d = self.stage5d(x5d)

        x4d = F.interpolate(x5d, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x4d = torch.cat((x4d, x4), dim=1)
        x4d = self.stage4d(x4d)

        x3d = F.interpolate(x4d, size=x3.size()[2:], mode='bilinear', align_corners=True)
        x3d = torch.cat((x3d, x3), dim=1)
        x3d = self.stage3d(x3d)

        x2d = F.interpolate(x3d, size=x2.size()[2:], mode='bilinear', align_corners=True)
        x2d = torch.cat((x2d, x2), dim=1)
        x2d = self.stage2d(x2d)

        x1d = F.interpolate(x2d, size=x1.size()[2:], mode='bilinear', align_corners=True)
        x1d = torch.cat((x1d, x1), dim=1)
        x1d = self.stage1d(x1d)

        # Output
        out = self.outconv(x1d)

        return out