"""
Training script, where training, validation, testing and saving are done
"""

import os
import torch.optim as optim
import torch
import torch.nn.functional as F
from modules import U2Net
from torch.utils.data import DataLoader
from dataset import load_data_helper
from predict import evaluate_model
from helper import get_base_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

default_num_classes = 6

def create_model(num_classes=default_num_classes):
    model = U2Net(in_ch=1, out_ch=num_classes).to(device)
    return model

def train_model(model: U2Net, train_loader: DataLoader, test_loader: DataLoader, num_epochs=10, num_classes=default_num_classes, goal: int=None):
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
        test_dice = evaluate_model(model, test_loader, num_classes)
        print(f"    test_dice: {test_dice:.4f}")

        if goal is not None and test_dice >= goal and epoch > 2:
            print(f"Goal reached: test_dice {test_dice:.4f} >= {goal}")
            break

def save_model(model: U2Net, path: str):
    torch.save(model.state_dict(), path)


if __name__ == "__main__":
    print(f"Using device: {device}")

    # load data
    root_dir = get_base_path()
    base_dir = os.path.join(root_dir, "keras_slices_data")
    total_classes = 6  # Fixed based on prior analysis  
    train_dataset, train_loader = load_data_helper(base_dir, "train", total_classes)
    test_dataset, test_loader = load_data_helper(base_dir, "test", total_classes)

    model = create_model(num_classes=total_classes)

    print("=> Starting training...")
    train_model(model, train_loader, test_loader, num_epochs=10, num_classes=total_classes, goal=0.75)

    save_model(model, f"{root_dir}/u2net_prostate_seg.pth")