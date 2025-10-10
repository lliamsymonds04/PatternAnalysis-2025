from ultralytics import YOLO
import os

# Compute absolute path of this model's folder and switch working dir so
# any downloaded model files and default 'runs' folder are created here
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = script_dir
os.chdir(base_dir)

# Load a model (instantiate after chdir so any downloads end up in base_dir)
model = YOLO("yolov8n.pt")  # for detection

# Ensure YOLO writes runs/checkpoints inside the model folder instead of repo root
output_dir = os.path.join(base_dir, "runs")
os.makedirs(output_dir, exist_ok=True)

# `project` controls the top-level folder for runs; `name` sets run subfolder.
model.train(data=os.path.join(base_dir, "data.yaml"), epochs=50, imgsz=480,
			batch=16, name="lesion-detector", project=output_dir, exist_ok=True)