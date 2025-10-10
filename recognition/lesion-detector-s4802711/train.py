from ultralytics import YOLO
import os

# Load a model
model = YOLO("yolov8n.pt") # for detection

# Train the model
base_dir = os.path.abspath("recognition/lesion-detector-s4802711")

model.train(data=os.path.join(base_dir,"data.yaml"), epochs=50, imgsz=480, batch=16, name="lesion-detector")