import os
import numpy as np
import cv2


def label_data(dir: str, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    mask = cv2.imread()
    for mask_name in os.listdir(dir):
        if not mask_name.endswith(".png"):
            continue

        mask_path = os.path.join(dir, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h,w = mask.shape


        with open(os.path.join(output_dir, mask_name.replace(".png", ".txt")), 'wb') as f:
            for cnt in contours:
                x,y,bw,bh = cv2.boundingRect(cnt)
                cx = x + bw/2
                cy = y + bh/2
                f.write(f"0 {cx/w} {cy/h} {w/w} {h/h}\n")

