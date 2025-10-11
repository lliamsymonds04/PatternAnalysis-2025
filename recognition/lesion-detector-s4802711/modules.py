from ultralytics import YOLO
import os

# model_name = "yolov8n.pt"  # nano
# model_name = "yolov8s.pt"  # small model
model_name = "yolov8m.pt"  # medium model
# model_name = "yolov8x.pt"  # better model but takes forever


def get_model():
    # any downloaded model files and default 'runs' folder are created here
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = script_dir
    os.chdir(base_dir)

    model = YOLO(model_name)

    return model