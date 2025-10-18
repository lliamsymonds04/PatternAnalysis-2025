import numpy as np
import nibabel as nib
from tqdm import tqdm
import os
import torch
from torch.utils.data import Dataset, DataLoader
import pathlib

# --- Utility Functions ---
def get_filenames(directory: str) -> list:
    """Get sorted list of NIfTI files in a directory."""
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.nii') or f.endswith('.nii.gz')]
    return files

def to_channels (arr: np.ndarray, total_classes: int, dtype=np.uint8) -> np.ndarray:
    """
    Converts a single-channel integer-labeled segmentation map to one-hot encoding,
    guaranteeing a fixed number of channels defined by total_classes.
    """
    arr_int = arr.astype(int)
    
    res = np.zeros(arr_int.shape + (total_classes,), dtype=dtype)
    
    unique_labels = np.unique(arr_int)
    
    for c_value in unique_labels:
        c_value = int(c_value)
        
        # Ensure the label value is within the expected range
        if c_value < total_classes:
            # The label value is used directly as the channel index (assuming labels are 0, 1, 2, ...)
            res[arr_int == c_value, c_value] = 1

    return res

class NiftiSegmentationDataset(Dataset):
    """
    A PyTorch Dataset for loading 2D slices from NIfTI files for segmentation tasks.
    """
    def __init__(self,
                 image_fnames: list,
                 norm_image: bool = False,
                 dtype=np.float32):

        self.image_fnames = image_fnames
        self.norm_image = norm_image
        self.dtype = dtype

    def __len__(self):
        return len(self.image_fnames)
        
    def _pad_or_crop(self, image: np.ndarray, target_shape: tuple) -> np.ndarray:
        """Pads or center-crops the 2D image/mask to the target shape (H, W)."""
        H, W = image.shape
        target_H, target_W = target_shape

        # 1. Padding (if smaller than target)
        pad_H = max(0, target_H - H)
        pad_W = max(0, target_W - W)
        
        # Calculate padding amounts to center the image
        pad_top = pad_H // 2
        pad_bottom = pad_H - pad_top
        pad_left = pad_W // 2
        pad_right = pad_W - pad_left
        
        if pad_H > 0 or pad_W > 0:
            image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
            H, W = image.shape # Update H, W after padding

        # 2. Cropping (if larger than target)
        crop_H = max(0, H - target_H)
        crop_W = max(0, W - target_W)

        # Calculate crop amounts to center the image
        crop_top = crop_H // 2
        crop_bottom = H - (crop_H - crop_top)
        crop_left = crop_W // 2
        crop_right = W - (crop_W - crop_left)
        
        if crop_H > 0 or crop_W > 0:
            image = image[crop_top:crop_bottom, crop_left:crop_right]
        
        # Ensure final shape is correct
        if image.shape != target_shape:
            raise ValueError(f"Reshaping failed. Got {image.shape}, expected {target_shape}")

        return image

    def _preprocess(self, inImage: np.ndarray, is_mask: bool = False) -> np.ndarray:
        """Applies common preprocessing steps."""

        if len(inImage.shape) == 3:
            inImage = inImage[:, :, inImage.shape[2] // 2]

        inImage = inImage.astype(self.dtype)

        TARGET_H, TARGET_W = 256, 256 # Define a safe target size
        inImage = self._pad_or_crop(inImage, (TARGET_H, TARGET_W))

        if not is_mask and self.norm_image:
            if inImage.std() != 0:
                inImage = (inImage - inImage.mean()) / inImage.std()
            else:
                inImage = inImage - inImage.mean()

        if is_mask and self.categorical_mask:
            inImage = to_channels(inImage, total_classes=self.total_classes, dtype=self.dtype)
        
        if not is_mask and len(inImage.shape) == 2:
            inImage = np.expand_dims(inImage, axis=-1)
        
        # Transpose to (C, H, W) for PyTorch
        if len(inImage.shape) == 3:
            inImage = np.transpose(inImage, (2, 0, 1))

        return inImage

    # def __getitem__(self, idx):
    #     image_nifti = nib.load(self.image_fnames[idx])
    #     image_data = image_nifti.get_fdata(caching='unchanged')
    #     image_tensor = self._preprocess(image_data, is_mask=False)

    #     image_tensor = torch.from_numpy(image_tensor)

    #     return image_tensor
    def __getitem__(self, idx):
        img = nib.load(self.image_fnames[idx]).get_fdata(caching='unchanged')
        if len(img.shape) == 3:
            img = img[:, :, 0]
        img = img.astype(self.dtype)
        img = (img - img.mean()) / (img.std() + 1e-8)
        img = np.expand_dims(img, axis=0)  # (1, H, W)
        return torch.from_numpy(img)
        
def load_data_helper(base_dir: str, subset: str):
    img_dir = os.path.join(base_dir, f"keras_slices_{subset}")

    img_files = get_filenames(img_dir)
    
    dataset = NiftiSegmentationDataset(
        img_files, 
        norm_image=True, 
        dtype=np.float32
    )
    
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    return dataset, loader

if __name__ == "__main__":
    root_dir = pathlib.Path(__file__).parent.resolve()
    base_dir = root_dir / "keras_slices_data"
    subset = "train"
    img_dir = base_dir / f"keras_slices_{subset}"
    print("image dir:", img_dir)

    if not img_dir.exists():
        print("Directory does not exist:", img_dir)
        print("Available directories under project path:")
        for p in sorted(root_dir.iterdir()):
            print("  ", p.name)
        raise FileNotFoundError(f"Data directory not found: {img_dir}")

    img_files = get_filenames(img_dir)
    
    test_dataset = NiftiSegmentationDataset(
        img_files,
        norm_image=True,
        dtype=np.float32
    )
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # Test an item to verify shapes
    first_image = test_dataset[0]
    
    print("-" * 50)
    print(f"Test image shape: {first_image.shape} (C, H, W)")
    print(f"Total samples in test dataset: {len(test_dataset)}")
    print(f"Total batches in test loader: {len(test_loader)}")