from ultralytics import YOLO

# Load the YOLOv8n model
model = YOLO('yolov8n.pt')

# Train the model
model.train(
    data=r"D:\automatic-number-plate-recognition-python-yolov8-main\dataset\data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device='cpu'
)
