from pathlib import Path
import os
import numpy as np
import torch
from torchvision.ops import box_iou
from ultralytics import YOLO
from PIL import Image
from helper import get_imgsz

base = Path(__file__).parent
data_yaml = base / "data.yaml"
val_images = base / "ISIC2018/dataset/images/val"
val_labels = base / "ISIC2018/dataset/labels/val"
runs_dir = base / "runs"
project_name = "lesion-detector"  # same name used during training
device = "cuda" if torch.cuda.is_available() else "cpu"
imgsz = get_imgsz()
conf = 0.001

def find_weights(runs_dir, project_name):
    # find best.pt under runs/<project>/<run>/weights/best.pt or last.pt
    if not runs_dir.exists():
        raise FileNotFoundError("runs directory not found")
    for run in sorted(runs_dir.glob(f"**/{project_name}/weights/*"), key=os.path.getmtime, reverse=True):
        if run.name in ("best.pt", "last.pt") or run.suffix == ".pt":
            return str(run)
    # fallback: any .pt under runs/<project_name>
    candidates = list(runs_dir.glob(f"**/{project_name}/**/*.pt"))
    return str(candidates[0]) if candidates else None

def xywhn_to_xyxy(x, y, w, h, W, H):
    xc = x * W
    yc = y * H
    bw = w * W
    bh = h * H
    xmin = xc - bw / 2
    ymin = yc - bh / 2
    xmax = xc + bw / 2
    ymax = yc + bh / 2
    return [xmin, ymin, xmax, ymax]

def read_yolo_label(path, img_w, img_h):
    boxes = []
    if not path.exists():
        return torch.empty((0,4))
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        # YOLO format: class x_center y_center width height (normalized)
        _, x, y, w, h = map(float, parts[:5])
        boxes.append(xywhn_to_xyxy(x, y, w, h, img_w, img_h))
    return torch.tensor(boxes, dtype=torch.float32) if boxes else torch.empty((0,4))

def main():
    weights = find_weights(runs_dir, project_name)
    if not weights:
        raise FileNotFoundError("No trained .pt weights found in runs/")
    print("Using weights:", weights)
    model = YOLO(weights)  # loads trained model
    model.to(device)

    img_paths = sorted(val_images.glob("*"))
    iou_per_image = []

    for img_path in img_paths:
        # find matching label file
        stem = img_path.stem
        label_path = val_labels / f"{stem}.txt"
        img = Image.open(img_path)
        W, H = img.size

        gt = read_yolo_label(label_path, W, H)  # (G,4)
        # run prediction
        preds = model.predict(source=str(img_path), imgsz=imgsz, conf=conf, device=device, verbose=False)
        r = preds[0]
        # try to extract xyxy from results (Ultralytics: r.boxes.xyxy)
        try:
            pred_boxes = r.boxes.xyxy.cpu() if hasattr(r.boxes, "xyxy") else r.boxes.data[:, :4].cpu()
        except Exception:
            # fallback empty
            pred_boxes = torch.empty((0,4))
        # ensure tensor
        pred_boxes = pred_boxes.detach() if isinstance(pred_boxes, torch.Tensor) else torch.tensor(pred_boxes, dtype=torch.float32)

        if gt.numel() == 0 and pred_boxes.numel() == 0:
            iou_per_image.append(1.0) 
            continue
        if gt.numel() == 0:
            iou_per_image.append(0.0)
            continue
        if pred_boxes.numel() == 0:
            iou_per_image.append(0.0)
            continue

        # compute IoU matrix and derive per-GT best IoU, then mean over GTs
        ious = box_iou(pred_boxes, gt)  # (P, G)
        # for each GT, take the best matching prediction
        best_per_gt, _ = ious.max(dim=0)
        mean_iou = float(best_per_gt.mean().item())
        iou_per_image.append(mean_iou)

    # summary
    iou_per_image = np.array(iou_per_image)
    print(f"Images evaluated: {len(iou_per_image)}")
    print(f"Mean IoU (mean over images): {iou_per_image.mean():.4f}")
    print(f"Median IoU: {np.median(iou_per_image):.4f}")
    print(f"Minimum IoU: {iou_per_image.min():.4f}")
    print(f"Images with IoU == 0: {(iou_per_image == 0).sum()}")

if __name__ == "__main__":
    main()