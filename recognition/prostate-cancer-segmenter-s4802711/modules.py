"""
Components for the 2D U-Net architecture.
"""

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
        return x
