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
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for images, masks_one_hot in train_loader:
            images = images.to(device)

            mask_indices = torch.argmax(masks_one_hot, dim=1).to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            loss = F.cross_entropy(outputs, mask_indices)

            # backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

def save_model(model: U2Net, path: str):
    torch.save(model.state_dict(), path)


if __name__ == "__main__":
    model = create_model()

    # load data
    base_dir = "recognition/prostate-cancer-segmenter-s4802711/keras_slices_data"
    base_dir = os.path.abspath(base_dir)
    dataset = load_data_helper(base_dir, "train")
    

    train_model(model, dataset, num_epochs=50)
    save_model(model, "u2net_prostate_seg.pth")