"""
Training script, where training, validation, testing and saving are done
"""

import os
import torch.optim as optim
import torch
import torch.nn.functional as F
from modules import U2Net
from dataset import load_data_helper
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_model():
    model = U2Net(in_ch=1, out_ch=4).to(device)
    return model

def dice_score(preds, targets, num_classes=4, epsilon=1e-6):
    dice_per_class = []
    for cls in range(num_classes):
        pred_cls = (preds == cls).float()
        target_cls = (targets == cls).float()
        
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()

        if union == 0:
            continue
        
        dice = (2. * intersection + epsilon) / (union + epsilon)
        dice_per_class.append(dice)

    if len(dice_per_class) == 0:
        return 0.0

    return sum(dice_per_class) / num_classes

@torch.no_grad()
def evaluate_model(model: U2Net, data_loader: DataLoader, num_classes=4):
    model.eval()
    total_dice = 0.0
    num_batches = 0
    
    for images, masks_one_hot in data_loader:
        images = images.to(device)
        mask_indices = torch.argmax(masks_one_hot, dim=1).to(device)

        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        
        dice = dice_score(preds, mask_indices, num_classes)
        total_dice += dice
        num_batches += 1
            
    return total_dice / num_batches if num_batches > 0 else 0.0

def train_model(model: U2Net, train_loader: DataLoader, test_loader: DataLoader, num_epochs=10, num_classes=4):
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

def save_model(model: U2Net, path: str):
    torch.save(model.state_dict(), path)


if __name__ == "__main__":
    model = create_model()

    print(f"Using device: {device}")

    # load data
    base_dir = "recognition/prostate-cancer-segmenter-s4802711/keras_slices_data"
    base_dir = os.path.abspath(base_dir)
    train_dataset, train_loader = load_data_helper(base_dir, "train")
    print(f"Images shape: {train_dataset.images.shape}, Masks shape: {train_dataset.masks.shape}")
    test_dataset, test_loader = load_data_helper(base_dir, "test")
    
    num_classes = train_dataset.masks.shape[1]

    train_model(model, train_loader, test_loader, num_epochs=10, num_classes=num_classes)
    save_model(model, "u2net_prostate_seg.pth")