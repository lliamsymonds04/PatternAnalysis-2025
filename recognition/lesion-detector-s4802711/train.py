from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt") # for detection

# Train the model
model.train(data="data.yaml", epochs=50, imgsz=480, batch=16, name="lesion-detector")