"""
Check all unique class values across the entire dataset
"""

import nibabel as nib
import numpy as np
import os
from tqdm import tqdm

base_dir = "recognition/prostate-cancer-segmenter-s4802711/keras_slices_data"
base_dir = os.path.abspath(base_dir)

all_unique_values = set()

for subset in ["train", "test", "validate"]:
    seg_dir = os.path.join(base_dir, f"keras_slices_seg_{subset}")
    
    if not os.path.exists(seg_dir):
        print(f"Skipping {subset} - directory doesn't exist")
        continue
    
    files = [f for f in os.listdir(seg_dir) if f.endswith('.nii') or f.endswith('.nii.gz')]
    
    print(f"\n{'='*60}")
    print(f"Checking {subset.upper()} set: {len(files)} files")
    print(f"{'='*60}")
    
    subset_values = set()
    
    for filename in tqdm(files[:100], desc=f"Scanning {subset}"):  # Check first 100 files
        filepath = os.path.join(seg_dir, filename)
        try:
            data = nib.load(filepath).get_fdata(caching='unchanged')
            unique_vals = np.unique(data)
            subset_values.update(unique_vals)
            all_unique_values.update(unique_vals)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    print(f"Unique values found in {subset}: {sorted(subset_values)}")

print(f"\n{'='*60}")
print(f"ALL UNIQUE VALUES ACROSS ENTIRE DATASET:")
print(f"{'='*60}")
print(f"Values: {sorted(all_unique_values)}")
print(f"Number of classes: {len(all_unique_values)}")

# Determine if classes are consecutive
sorted_values = sorted(all_unique_values)
expected_consecutive = list(range(int(sorted_values[0]), int(sorted_values[-1]) + 1))

if sorted_values == [float(x) for x in expected_consecutive]:
    print(f"\n✓ Classes are consecutive: {sorted_values}")
    print(f"  No remapping needed - use num_classes={len(sorted_values)}")
else:
    print(f"\n⚠ Classes are NOT consecutive!")
    print(f"  Found: {sorted_values}")
    print(f"  Expected (if consecutive): {expected_consecutive}")
    print(f"  Need to remap to: {list(range(len(sorted_values)))}")
    print(f"  Use num_classes={len(sorted_values)}")
