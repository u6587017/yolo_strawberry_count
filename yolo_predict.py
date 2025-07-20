from ultralytics import YOLO

model = YOLO('runs/detect/train/weights/best.pt')
result = model.predict('test.mp4', save=True, conf=0.5)