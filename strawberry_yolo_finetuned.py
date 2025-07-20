from ultralytics import YOLO
def main():
    model = YOLO("yolov8n.pt")  # หรือ yolov8s.pt ถ้ามึงใช้ตัวอื่น
    results = model.train(data="strawberry.yml", epochs=100, imgsz=640, device="cuda")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # เผื่อไว้ถ้า build exe ในอนาคต
    main()