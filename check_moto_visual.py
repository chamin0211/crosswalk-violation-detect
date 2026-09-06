"""
낮은 conf에서 잡히는 오토바이 검출이 실제 오토바이인지 눈으로 확인한다.
숫자만으로는 진짜와 오탐을 구분할 수 없다.
"""
import cv2
from ultralytics import YOLO

VIDEO = "data/raw/right_road.mp4"
OUT = "data/moto_check.mp4"
CONF = 0.15
IMGSZ = 1920

model = YOLO("yolo11m.pt")
cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

writer = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

frame_idx = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break

    r = model.predict(frame, device=0, conf=CONF, imgsz=IMGSZ,
                      classes=[3], verbose=False)[0]

    for i in range(len(r.boxes)):
        score = float(r.boxes.conf[i].item())
        x1, y1, x2, y2 = [int(v) for v in r.boxes.xyxy[i].tolist()]
        # conf에 따라 색을 다르게: 높으면 초록, 낮으면 빨강
        if score >= 0.35:
            color = (0, 255, 0)
        elif score >= 0.25:
            color = (0, 255, 255)
        else:
            color = (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame, "moto {:.2f}".format(score), (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    writer.write(frame)
    frame_idx += 1
    if frame_idx % 300 == 0:
        print("  진행: {}/{}".format(frame_idx, total))

cap.release()
writer.release()
print("완료:", OUT)
