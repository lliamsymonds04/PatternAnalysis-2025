"""
Script to run inference using a trained prostate cancer segmentation model.
"""

import os
import torch
from modules import U2Net
import nibabel as nib
from dataset import load_data_helper
from torch.utils.data import DataLoader
from helper import get_base_path

model_name = "u2net_prostate_seg.pth"
model_path = os.path.join(get_base_path(), model_name)

# Load the model
model = U2Net(in_ch=1, out_ch=6)
model.load_state_dict(torch.load(model_path))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model loaded: {model_name}, using device: {device}")

# load the test data
root_dir = "recognition/prostate-cancer-segmenter-s4802711"
base_dir = os.path.join(root_dir, "keras_slices_data")
base_dir = os.path.abspath(base_dir)
total_classes = 6  # Fixed based on prior analysis
test_dataset, test_loader = load_data_helper(base_dir, "test", total_classes)

#check the dice score
def dice_score(preds, targets, num_classes=total_classes, epsilon=1e-6):
    dice_per_class = []
    for cls in range(num_classes):
        pred_cls = (preds == cls).float()
        target_cls = (targets == cls).float()
        
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()

        if cls == 0:
            # Skip background class
            continue

        if union == 0:
            # If this class doesn't appear in ground truth or prediction
            dice_per_class.append(1.0)  # Perfect score for absent class
        else:
            dice = (2. * intersection + epsilon) / (union + epsilon)
            dice_per_class.append(dice)

    if len(dice_per_class) == 0:
        return 0.0

    return sum(dice_per_class) / (num_classes - 1)  # exclude background

@torch.no_grad()
def evaluate_model(model: U2Net, data_loader: DataLoader, num_classes=total_classes):
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

test_dice = evaluate_model(model, test_loader, total_classes)
print(f"Test Dice Score: {test_dice:.4f}")
