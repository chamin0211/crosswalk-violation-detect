"""
본인 CCTV 영상 -> PCPA 입력 4종 생성 -> 확률 예측
1. YOLO11로 보행자 검출 + 추적 (ByteTrack)
2. 특정 보행자 트랙에서 연속 16프레임 샘플링
3. local_context(112x112 crop), pose(34), box(4), speed(1) 생성
4. PCPA 모델로 확률 예측
"""

import os
import sys
import numpy as np
import cv2
from ultralytics import YOLO

# PCPA 모델 로드를 위한 경로 설정
sys.path.insert(0, os.path.expanduser("~/DPJI/Intention"))
sys.path.insert(0, os.path.expanduser("~/DPJI/Intention/extras"))

VIDEO_PATH = "여기에_CCTV_영상_경로.mp4"
SEQ_LEN = 16
CROP_SIZE = 112

# --- 1. YOLO11로 검출+추적 ---
yolo = YOLO("yolo11n.pt")
results = yolo.track(source=VIDEO_PATH, classes=[0], persist=True, device="cpu")

# --- 2. 트랙별로 프레임 모으기 ---
tracks = {}  # track_id -> list of (frame_img, bbox, keypoints)

pose_model = YOLO("yolo11n-pose.pt")
pose_results = pose_model.track(source=VIDEO_PATH, classes=[0], persist=True, device="cpu")

cap = cv2.VideoCapture(VIDEO_PATH)
frame_idx = 0

for det_r, pose_r in zip(results, pose_results):
    ret, frame = cap.read()
    if not ret:
        break

    if det_r.boxes is not None and det_r.boxes.id is not None:
        for i, track_id in enumerate(det_r.boxes.id.tolist()):
            box = det_r.boxes.xyxy[i].tolist()  # x1,y1,x2,y2

            # pose에서 같은 track_id 찾기 (간단 매칭, 실제로는 IoU 매칭 권장)
            kpts = None
            if pose_r.keypoints is not None and pose_r.boxes.id is not None:
                for j, pid in enumerate(pose_r.boxes.id.tolist()):
                    if pid == track_id:
                        kpts = pose_r.keypoints.xy[j].flatten().tolist()
                        break
            if kpts is None or len(kpts) != 34:
                kpts = [0.0] * 34  # pose 못 찾으면 0으로 채움

            tracks.setdefault(track_id, []).append({
                "frame": frame.copy(),
                "box": box,
                "pose": kpts,
            })

    frame_idx += 1

cap.release()

print(f"검출된 트랙 수: {len(tracks)}")
for tid, seq in tracks.items():
    print(f"  track {tid}: {len(seq)}프레임")

# --- 3. 16프레임 이상인 트랙만 사용, local_context/box/pose/speed 생성 ---
def build_inputs(seq):
    seq16 = seq[-SEQ_LEN:] if len(seq) >= SEQ_LEN else seq + [seq[-1]] * (SEQ_LEN - len(seq))

    local_context = []
    box_seq = []
    pose_seq = []
    speed_seq = []

    for item in seq16:
        x1, y1, x2, y2 = item["box"]
        h, w = item["frame"].shape[:2]
        x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
        crop = item["frame"][y1:y2, x1:x2]
        if crop.size == 0:
            crop = np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
        else:
            crop = cv2.resize(crop, (CROP_SIZE, CROP_SIZE))
        local_context.append(crop / 255.0)

        box_seq.append(item["box"])
        pose_seq.append(item["pose"])
        speed_seq.append([0.0])  # 고정 CCTV라 실제 속도 없음 -> 임시 0

    return (
        np.array(local_context, dtype=np.float32)[None, ...],   # (1,16,112,112,3)
        np.array(pose_seq, dtype=np.float32)[None, ...],          # (1,16,34)
        np.array(box_seq, dtype=np.float32)[None, ...],           # (1,16,4)
        np.array(speed_seq, dtype=np.float32)[None, ...],         # (1,16,1)
    )

# --- 4. 가장 긴 트랙 하나로 예측 테스트 ---
from tensorflow.keras.models import load_model

model = load_model(
    "/home/aiuser/DPJI/checkpoints/intention/pcpa/13Sep2024-21h06m20s/model.h5",
    compile=False,
)

longest_track = max(tracks.items(), key=lambda kv: len(kv[1]))
tid, seq = longest_track
print(f"\n예측 대상 track {tid} ({len(seq)}프레임)")

local_ctx, pose_arr, box_arr, speed_arr = build_inputs(seq)

pred = model.predict([local_ctx, pose_arr, box_arr, speed_arr])
print(f"\n횡단 의도 확률: {pred[0][0]:.4f}")