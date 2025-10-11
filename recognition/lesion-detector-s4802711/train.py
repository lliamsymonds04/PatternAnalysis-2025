from ultralytics import YOLO
import os
import multiprocessing
from helper import get_imgsz, get_project_name

model_name = "yolov8n.pt"  # nano
# model_name = "yolov8s.pt"  # small model
# model_name = "yolov8x.pt"  # better model but takes forever


def main():
	# any downloaded model files and default 'runs' folder are created here
	script_dir = os.path.dirname(os.path.abspath(__file__))
	base_dir = script_dir
	os.chdir(base_dir)

	model = YOLO(model_name)

	# Ensure YOLO writes runs/checkpoints inside the model folder instead of repo root
	output_dir = os.path.join(base_dir, "runs")
	os.makedirs(output_dir, exist_ok=True)

	model.train(data=os.path.join(base_dir, "data.yaml"), epochs=50, imgsz=get_imgsz(),
				batch=16, name=get_project_name(), project=output_dir, exist_ok=True)


if __name__ == "__main__":
	multiprocessing.freeze_support()
	main()