"""
Components for the 2D U-Net architecture.
"""

import torch.nn as nn
import torch.nn.functional as F

class RSU(nn.m):
    def __init__(self, in_ch, out_ch, mid_ch, depth):
        super(RSU, self).__init__()
        self.depth = depth