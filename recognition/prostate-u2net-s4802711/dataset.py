import numpy as np
import nibabel as nib
from tqdm import tqdm
import os
import torch
from torch.utils.data import Dataset, DataLoader
from helper import get_base_path

# --- Utility Functions ---


def get_filenames(directory: str) -> list:
    """Get sorted list of NIfTI files in a directory."""
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.nii') or f.endswith('.nii.gz')]
    return files

def get_total_classes(seg_files: list[str]) -> int:
    """
    Scans all segmentation NIfTI files to find the maximum label value.
    Total classes = max_label + 1 (to include label 0/background).
    """
    max_label = 0
    print(f"Scanning {len(seg_files)} segmentation files to determine total number of classes...")
    for seg_file in tqdm(seg_files):
        try:
            seg_data = nib.load(seg_file).get_fdata(caching='unchanged')
            
            if len(seg_data.shape) == 3:
                seg_data = seg_data[:, :, 0]

            unique_labels = np.unique(seg_data.astype(int))
            if len(unique_labels) > 0:
                current_max = np.max(unique_labels)
                if current_max > max_label:
                    max_label = current_max
                    
        except Exception as e:
            print(f"Warning: Could not process file {seg_file}. Error: {e}")
            
    total_classes = int(max_label) + 1
    print(f"Total number of classes determined to be: {total_classes} (Max label found was {max_label})")
    return total_classes

def to_channels (arr: np.ndarray, total_classes: int, dtype=np.uint8) -> np.ndarray:
    """
    Converts a single-channel integer-labeled segmentation map to one-hot encoding,
    guaranteeing a fixed number of channels defined by total_classes.
    """
    arr_int = arr.astype(int)
    
    # FIX: Use the fixed total_classes for output shape, not len(channels)
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
                 mask_fnames: list, 
                 total_classes: int, # <-- NEW REQUIRED ARGUMENT
                 norm_image: bool = False,
                 categorical_mask: bool = True,
                 dtype=np.float32):

        if len(image_fnames) != len(mask_fnames):
            raise ValueError("Image and mask file lists must have the same length.")

        self.image_fnames = image_fnames
        self.mask_fnames = mask_fnames
        self.total_classes = total_classes # <-- STORED
        self.norm_image = norm_image
        self.categorical_mask = categorical_mask
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
        # Calculate padding amounts to center the image
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
            image = np.pad(
                image,
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                mode="constant",
                constant_values=0,
            )
            H, W = image.shape  # Update H, W after padding

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
            raise ValueError(
                f"Reshaping failed. Got {image.shape}, expected {target_shape}"
            )

        return image

    def _preprocess(self, inImage: np.ndarray, is_mask: bool = False) -> np.ndarray:
        """Applies common preprocessing steps."""

        if len(inImage.shape) == 3:
            inImage = inImage[:, :, 0]

        inImage = inImage.astype(self.dtype)

        TARGET_H, TARGET_W = 256, 256  # Define a safe target size
        inImage = self._pad_or_crop(inImage, (TARGET_H, TARGET_W))

        if not is_mask and self.norm_image:
            if inImage.std() != 0:
                inImage = (inImage - inImage.mean()) / inImage.std()
            else:
                inImage = inImage - inImage.mean()

        if is_mask and self.categorical_mask:
            inImage = to_channels(
                inImage, total_classes=self.total_classes, dtype=self.dtype
            )

        if not is_mask and len(inImage.shape) == 2:
            inImage = np.expand_dims(inImage, axis=-1)

        # Transpose to (C, H, W) for PyTorch
        if len(inImage.shape) == 3:
            inImage = np.transpose(inImage, (2, 0, 1))

        return inImage

    def __getitem__(self, idx):
        image_nifti = nib.load(self.image_fnames[idx])
        image_data = image_nifti.get_fdata(caching="unchanged")
        image_tensor = self._preprocess(image_data, is_mask=False)

        mask_nifti = nib.load(self.mask_fnames[idx])
        mask_data = mask_nifti.get_fdata(caching="unchanged")
        mask_tensor = self._preprocess(mask_data, is_mask=True)

        image_tensor = torch.from_numpy(image_tensor)
        mask_tensor = torch.from_numpy(mask_tensor)

        return image_tensor, mask_tensor


def load_data_helper(base_dir: str, subset: str, total_classes: int):
    img_dir = os.path.join(base_dir, f"keras_slices_{subset}")
    seg_dir = os.path.join(base_dir, f"keras_slices_seg_{subset}")

    img_files = get_filenames(img_dir)
    seg_files = get_filenames(seg_dir)

    dataset = NiftiSegmentationDataset(
        img_files,
    total_classes = get_total_classes(seg_files)
    
    test_dataset = NiftiSegmentationDataset(
        img_files,
        seg_files,
        total_classes=total_classes,  # Pass the fixed count
        norm_image=True,
        categorical_mask=True, 
        dtype=np.float32
    )
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # Test an item to verify shapes
    first_image, first_mask = test_dataset[0]
    
    print("-" * 50)
    print(f"Total determined classes: {test_dataset.total_classes}")
    print(f"Test image shape: {first_image.shape} (C, H, W)")
    print(f"Test mask shape: {first_mask.shape} (C, H, W)")
    
    # Verify the channel size matches the fixed count
    assert first_mask.shape[0] == total_classes, "Mask channel count mismatch!"
    
    masks_labels = first_mask.argmax(dim=0)
    print(f"Unique mask values in first sample: {torch.unique(masks_labels)}")
    print(f"Total samples in test dataset: {len(test_dataset)}")
    print(f"Total batches in test loader: {len(test_loader)}")
