"""
영상 전체를 훑으며 클래스별 검출 신뢰도 분포를 측정한다.
목적:
  1. 오토바이가 어느 conf에서 잡히는지 (임계값 근거 확보)
  2. conf를 낮췄을 때 오탐이 얼마나 늘어나는지
낮은 conf(0.05)로 한 번만 돌려서 전부 수집한 뒤, 사후에 필터링해서 비교한다.
"""
import cv2
import numpy as np
from ultralytics import YOLO

VIDEO = "data/raw/moto_20s.mp4"
SCAN_CONF = 0.05          # 일단 낮게 잡아 전부 수집
IMGSZ = 1920
TARGET_CLASSES = [0, 1, 2, 3, 5, 7]

CLASS_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

model = YOLO("yolo11m.pt")
cap = cv2.VideoCapture(VIDEO)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("총 프레임:", total_frames)
print("스캔 시작 (conf={})".format(SCAN_CONF))
print("=" * 60)

# 클래스별 conf 값을 전부 모은다
conf_by_class = {}
for c in TARGET_CLASSES:
    conf_by_class[c] = []

frame_idx = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break

    r = model.predict(frame, device=0, conf=SCAN_CONF, imgsz=IMGSZ,
                      classes=TARGET_CLASSES, verbose=False)[0]

    for i in range(len(r.boxes)):
        cls_id = int(r.boxes.cls[i].item())
        score = float(r.boxes.conf[i].item())
        conf_by_class[cls_id].append(score)

    frame_idx += 1
    if frame_idx % 300 == 0:
        print("  진행: {}/{}".format(frame_idx, total_frames))

cap.release()

print("=" * 60)
print("클래스별 검출 신뢰도 분포")
print("=" * 60)

for c in TARGET_CLASSES:
    vals = conf_by_class[c]
    name = CLASS_NAMES[c]
    if len(vals) == 0:
        print("{:12s} 검출 없음".format(name))
        continue
    arr = np.array(vals)
    print("{:12s} 총 {}회".format(name, len(arr)))
    print("             최소 {:.3f} / 중앙값 {:.3f} / 최대 {:.3f}".format(
        arr.min(), np.median(arr), arr.max()))
    for th in [0.35, 0.30, 0.25, 0.20, 0.15, 0.10]:
        kept = int((arr >= th).sum())
        pct = 100.0 * kept / len(arr)
        print("             conf {:.2f} 이상: {:5d}회 ({:5.1f}%)".format(th, kept, pct))
    print()

print("=" * 60)
print("임계값별 프레임당 평균 검출 수 (오탐 추정용)")
print("=" * 60)

for th in [0.35, 0.30, 0.25, 0.20, 0.15, 0.10]:
    total = 0
    for c in TARGET_CLASSES:
        arr = np.array(conf_by_class[c]) if conf_by_class[c] else np.array([])
        if len(arr) > 0:
            total += int((arr >= th).sum())
    per_frame = total / frame_idx if frame_idx > 0 else 0
    print("conf {:.2f}: 총 {:6d}개, 프레임당 {:.2f}개".format(th, total, per_frame))

print("=" * 60)
