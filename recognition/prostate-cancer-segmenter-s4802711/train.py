"""
Training script, where training, validation, testing and saving are done
"""

import os
import torch.optim as optim
import torch
import torch.nn.functional as F
from modules import U2Net
from dataset import SegmentationDataset, load_data_helper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_model():
    model = U2Net(in_ch=1, out_ch=4).to(device)
    return model

def train_model(model: U2Net, train_loader: SegmentationDataset, num_epochs=10):
    pass


if __name__ == "__main__":
    model = create_model()

    # load data
    base_dir = "recognition/prostate-cancer-segmenter-s4802711/keras_slices_data"
    base_dir = os.path.abspath(base_dir)
    dataset = load_data_helper(base_dir, "train")
    

    train_model(model, dataset, num_epochs=50)