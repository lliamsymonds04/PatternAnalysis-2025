import os
import shutil
import numpy as np
import cv2
import yaml
from tqdm import tqdm
import random
from pathlib import Path

def drop_segmentation_str(str: str) -> str:
    if str.endswith("_segmentation.png"):
        return str[:-17] + ".png"
    return str

def label_data(dir: str, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for mask_name in tqdm(os.listdir(dir)):
        if not mask_name.endswith(".png"):
            continue

        mask_path = os.path.join(dir, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h,w = mask.shape


        mask_name = drop_segmentation_str(mask_name)
        with open(os.path.join(output_dir, mask_name.replace(".png", ".txt")), 'w') as f:
            for cnt in contours:
                x,y,bw,bh = cv2.boundingRect(cnt)
                cx = x + bw/2
                cy = y + bh/2
                f.write(f"0 {cx/w} {cy/h} {bw/w} {bh/h}\n")

def prepare_dataset(output_dir: str, label_dir: str, images_dir:str, seed: int = 42):
    random.seed(seed)

    dataset_dir = os.path.join(output_dir, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    
    # create subfolders
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(dataset_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, "labels", split), exist_ok=True)

    # get all image files
    image_files = [f for f in os.listdir(images_dir) if f.endswith(".jpg") or f.endswith(".png")]
    random.shuffle(image_files)
    print(f"Total images found: {len(image_files)}")

    n = len(image_files)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)

    splits = {
        "train": image_files[:n_train],
        "val": image_files[n_train:n_train+n_val],
        "test": image_files[n_train+n_val:]
    }

    for split, file_list in splits.items():
        for file_name in file_list:
            # Image
            src_img = os.path.join(images_dir, file_name)
            dst_img = os.path.join(dataset_dir, "images", split, file_name)
            shutil.copy2(src_img, dst_img)

            # Label (same name, .txt)
            file_name = drop_segmentation_str(file_name)
            label_name = file_name.replace(".png", ".txt")
            src_label = os.path.join(label_dir, label_name)
            dst_label = os.path.join(dataset_dir, "labels", split, label_name)
            if os.path.exists(src_label):
                shutil.copy2(src_label, dst_label)

    print("Dataset preparation complete.")

def make_yolo_path(p: Path) -> str:
    return str(p.as_posix())
    
def create_yaml():
    Root = Path(__file__).parent.resolve()
    yaml_content = { 
        "train": make_yolo_path(Root / "ISIC2018" / 'dataset' / 'images' / 'train'),
        "val": make_yolo_path(Root / "ISIC2018" / 'dataset' / 'images' / 'val'),
        "test": make_yolo_path(Root / "ISIC2018" / 'dataset' / 'images' / 'test'),
        "nc": 1,
        "names": ['lesion']
    }

    yaml_path = Root / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f)

if __name__ == "__main__":
    base_dir = "recognition/lesion-detector-s4802711/ISIC2018"
    base_dir = os.path.abspath(base_dir)

    # training data
    ground_truth_dir = os.path.join(base_dir, "ISIC2018_Task1_Training_GroundTruth_x2")
    labels_output = os.path.join(base_dir, "labels_output")
    label_data(ground_truth_dir, labels_output)

    training_images_dir = os.path.join(base_dir, "ISIC2018_Task1-2_Training_Input_x2")
    test_images_dir = os.path.join(base_dir, "ISIC2018_Task1_Test_Input")

    prepare_dataset(base_dir, labels_output, training_images_dir)

    # create data.yaml for YOLOv8
    create_yaml()
