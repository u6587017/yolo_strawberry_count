import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")
cap = cv2.VideoCapture("test.mp4")

seen_centroids = []
total_strawberries = 0
DIST_THRESHOLD = 30  # ถ้าจุดใหม่ห่างจากจุดเดิมมาก → ถือว่าเป็นลูกใหม่

def is_new_centroid(centroid, seen_centroids, threshold=30):
    for old in seen_centroids:
        if np.linalg.norm(np.array(centroid) - np.array(old)) < threshold:
            return False
    return True

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # ตรวจด้วย YOLO
    results = model(frame)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        # 👇 ถ้า label == "strawberry" ถึงจะนับ (ใส่ชื่อ class ให้ตรงกับที่เทรน)
        if label.lower() == "strawberry":
            # ดึง bbox
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            centroid = (cx, cy)

            # นับลูกใหม่
            if is_new_centroid(centroid, seen_centroids, DIST_THRESHOLD):
                total_strawberries += 1
                seen_centroids.append(centroid)

            # วาด box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.circle(frame, centroid, 4, (255,0,0), -1)
            cv2.putText(frame, f"{label}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # ใส่ count ทั้งหมด
    cv2.putText(frame, f"Total: {total_strawberries}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("YOLO Strawberry Counter", frame)
    if cv2.waitKey(15) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
