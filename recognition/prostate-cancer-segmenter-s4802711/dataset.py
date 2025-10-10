"""
Dataloader and preprocessing for prostate cancer segmentation task.
"""

import os
import numpy as np
import nibabel as nib
from tqdm import tqdm

def get_filenames(dir: str):
    """Get list of NIfTI filenames in a directory."""
    files = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith('.nii') or f.endswith('.nii.gz')]
    return files

def load_data_helper(base_dir: str, subset: str):
    imgs_dir = os.path.join(base_dir, f"keras_slices_{subset}")
    segs_dir = os.path.join(base_dir, f"keras_slices_seg_{subset}")

    img_files = get_filenames(imgs_dir)
    seg_files = get_filenames(segs_dir)
    
    imgs = load_data_2D(img_files, normImage=True, categorical=False, dtype=np.float32,
                     getAffines=False, early_stop=False)
    segs = load_data_2D(seg_files, normImage=False, categorical=True, dtype=np.uint8,
                     getAffines=False, early_stop=False)

    return imgs, segs
    

def to_channels(arr: np.ndarray, dtype: np.uint8 = np.uint8) -> np.ndarray:
    """Convert a 2D label array into one-hot channel format.

    The function finds unique label values and creates a channel per label.
    """
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c:c + 1][arr == c] = 1

    return res

def load_data_2D(imageNames: list[str], normImage=False, categorical=False, dtype=np.float32,
                    getAffines=False, early_stop=False):
    """
    Load medical image data from a list of filenames.

    This function pre-allocates arrays for conv2d to avoid excessive memory usage.

    Parameters
    ----------
    imageNames : Sequence[str]
        Paths to NIfTI image files.
    normImage : bool
        If True, normalize each image to zero mean and unit std.
    categorical : bool
        If True, convert labels to one-hot channels.
    dtype : numpy dtype
        Output array dtype.
    getAffines : bool
        If True, return (images, affines) where affines is a list of image affines.
    early_stop : bool
        If True, stop after ~20 images (useful for quick tests).
    """
    affines = []

    # determine fixed size from first case
    num = len(imageNames)
    first_case = nib.load(imageNames[0]).get_fdata(caching='unchanged')
    if len(first_case.shape) == 3:
        # sometimes extra dims, remove singleton third dimension
        first_case = first_case[:, :, 0]

    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, channels = first_case.shape
        images = np.zeros((num, rows, cols, channels), dtype=dtype)
    else:
        rows, cols = first_case.shape
        images = np.zeros((num, rows, cols), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged')  # read from disk only
        affine = niftiImage.affine

        if len(inImage.shape) == 3:
            # sometimes extra dims in HipMRI_study data
            inImage = inImage[:, :, 0]

            inImage = inImage.astype(dtype)

            if normImage:
                # previous experiments used norm or scaling; here use z-score
                # inImage = inImage / np.linalg.norm(inImage)
                # inImage = 255. * inImage / inImage.max()
                inImage = (inImage - inImage.mean()) / inImage.std()

            if categorical:
                inImage = to_channels(inImage, dtype=dtype)
                images[i, :, :, :] = inImage
            else:
                images[i, :, :] = inImage

            affines.append(affine)

            if i > 20 and early_stop:
                break

    if getAffines:
        return images, affines
    else:
        return images

if __name__ == "__main__":
    base_dir = "recognition/prostate-cancer-segmenter-s4802711/keras_slices_data"
    base_dir = os.path.abspath(base_dir)
    train_imgs, train_segs = load_data_helper(base_dir, "train")
    print(f"Train images shape: {train_imgs.shape}, Train segs shape: {train_segs.shape}")