"""
YOLO11-pose 인퍼런스 테스트
- 보행자 17개 관절 좌표 추출
"""

import os
from ultralytics import YOLO

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "runs", "pose")

MODEL_NAME = "yolo11n-pose.pt"   # CPU 기준
VIDEO_PATH = os.path.join(PROJECT_ROOT, "data", "test_crosswalk.mp4")

model = YOLO(MODEL_NAME)

results = model.predict(
    source=VIDEO_PATH,
    conf=0.4,
    device="cpu",
    save=True,
    project=OUTPUT_DIR,
    name="crosswalk_pose_test",
)

for i, r in enumerate(results):
    kpts = r.keypoints
    if kpts is not None and kpts.xy.shape[0] > 0:
        n_people = kpts.xy.shape[0]
        print(f"frame {i}: {n_people}명 pose 검출")