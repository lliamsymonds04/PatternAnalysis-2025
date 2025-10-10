"""
Components for the 2D U-Net architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class RSU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, mid_ch: int, depth: int = 7):
        super(RSU, self).__init__()
        self.depth = depth

        self.rebnconvin = self._make_conv_block(in_ch, out_ch)

        # encoder stages
        self.rebnconv1 = self._make_conv_block(out_ch, mid_ch)
        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebconvs = nn.ModuleList()
        for i in range(2, depth):
            self.rebconvs.append(self._make_conv_block(mid_ch, mid_ch))

        self.rebconvm = self._make_conv_block(mid_ch, mid_ch)

        # decoder stages
        self.rebconvd = nn.ModuleList()
        for i in range(2, depth):
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
