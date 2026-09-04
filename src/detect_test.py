"""
YOLO11 사람/차량 검출 인퍼런스 테스트
- COCO 사전학습 가중치 사용
- 영상 파일 경로 하나 넣으면 바로 결과 영상 저장됨
"""

from ultralytics import YOLO

MODEL_NAME = "yolo11n.pt"    # CPU용 경량 모델로 변경
# MODEL_NAME = "yolo11s.pt"          # GPU(RTX 4060)니까 s 모델로
VIDEO_PATH = "data/test_crosswalk.mp4"   # 테스트 영상 경로
OUTPUT_DIR = "runs/detect"

model = YOLO(MODEL_NAME)
"yolo11n.pt" 
# 0=person, 2=car, 3=motorcycle, 5=bus, 7=truck
TARGET_CLASSES = [0, 2, 3, 5, 7]

results = model.predict(
    source=VIDEO_PATH,
    classes=TARGET_CLASSES,
    conf=0.4,
    device="cpu",             # device=0 대신 이렇게
    # device=0,          # GPU 사용
    save=True,
    project=OUTPUT_DIR,
    name="crosswalk_test",
)

for i, r in enumerate(results):
    boxes = r.boxes
    if boxes is not None and len(boxes) > 0:
        classes = [model.names[int(c)] for c in boxes.cls]
        print(f"frame {i}: {len(boxes)}개 검출 -> {classes}")