# debug_detect.py
import cv2
from ultralytics import YOLO

VIDEO = "data/uploaded_video.mp4"   # 실제 업로드된 영상 경로

yolo = YOLO("yolo11m.pt")

# 특정 프레임들만 직접 읽어서 검출
cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)

for sec in [11, 12, 13, 14, 15, 16]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps))
    ret, frame = cap.read()
    if not ret:
        continue
    r = yolo.predict(frame, classes=[0, 2, 3, 5, 7], conf=0.1,
                     device=0, verbose=False, imgsz=1920)[0]
    print(f"\n=== {sec}초 (프레임 {int(sec*fps)}) ===")
    if r.boxes is None or len(r.boxes) == 0:
        print("  검출 없음")
        continue
    for i in range(len(r.boxes)):
        cls = int(r.boxes.cls[i].item())
        conf = float(r.boxes.conf[i].item())
        box = [round(v) for v in r.boxes.xyxy[i].tolist()]
        print(f"  cls={cls} conf={conf:.3f} box={box}")

cap.release()