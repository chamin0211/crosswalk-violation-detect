"""
오토바이가 검출되는 conf 임계값을 찾는다.
어느 지점에서 잡히기 시작하는지 확인해서 원인을 특정한다.
"""
import cv2
from ultralytics import YOLO

VIDEO = "data/raw/right_road.mp4"
TARGET_SEC = 75.0   # 오토바이 두 대가 보이는 시점

CLASS_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
cap.set(cv2.CAP_PROP_POS_FRAMES, int(TARGET_SEC * fps))
ok, frame = cap.read()
cap.release()

if not ok:
    print("프레임 읽기 실패")
    raise SystemExit

print("프레임 크기:", frame.shape)
print("=" * 60)

model = YOLO("yolo11m.pt")

for conf_val in [0.35, 0.25, 0.15, 0.08, 0.03]:
    r = model.predict(frame, device=0, conf=conf_val, imgsz=1920,
                      classes=[0, 1, 2, 3, 5, 7], verbose=False)[0]
    moto_count = 0
    print("conf =", conf_val, "→ 총", len(r.boxes), "개 검출")
    for i in range(len(r.boxes)):
        cls_id = int(r.boxes.cls[i].item())
        score = float(r.boxes.conf[i].item())
        box = [round(v) for v in r.boxes.xyxy[i].tolist()]
        name = CLASS_NAMES.get(cls_id, str(cls_id))
        if cls_id in (1, 3):
            moto_count += 1
            print("   [오토바이/자전거]", name, "conf={:.3f}".format(score), box)
    print("   오토바이 검출 수:", moto_count)
    print()