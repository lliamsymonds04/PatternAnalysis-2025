"""
Training script, where training, validation, testing and saving are done
"""

import torch.optim as optim
import torch
import torch.nn.functional as F
from modules import U2Net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_model():
    model = U2Net(in_ch=1, out_ch=4).to(device)
    return model

def train_model(model: U2Net):
    pass  # Training logic to be implemented here