"""
uv 환경에서 실행: YOLO detect + pose로 트랙별 특징 뽑아서 npz로 저장
메모리 절약을 위해 stream=True로 프레임 하나씩 처리
"""
import numpy as np
import cv2
from ultralytics import YOLO

VIDEO_PATH = "data/test_crosswalk.mp4"
OUTPUT_NPZ = "data/track_features.npz"
CROP_SIZE = 112
MAX_FRAMES_PER_TRACK = 32

yolo = YOLO("yolo11n.pt")
pose_model = YOLO("yolo11n-pose.pt")

cap = cv2.VideoCapture(VIDEO_PATH)
tracks = {}
frame_idx = 0

det_gen = yolo.track(source=VIDEO_PATH, classes=[0], persist=True, device="cpu", stream=True, verbose=False)

for det_r in det_gen:
    ret, frame = cap.read()
    if not ret:
        break

    if det_r.boxes is not None and det_r.boxes.id is not None:
        pose_r = pose_model.predict(frame, device="cpu", verbose=False)[0]

        for i, track_id in enumerate(det_r.boxes.id.tolist()):
            box = det_r.boxes.xyxy[i].tolist()

            kpts = None
            if pose_r.keypoints is not None and len(pose_r.keypoints.xy) > 0:
                kpts = pose_r.keypoints.xy[0].flatten().tolist()
            if kpts is None or len(kpts) != 34:
                kpts = [0.0] * 34

            x1, y1, x2, y2 = box
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
            crop = frame[y1:y2, x1:x2]
            crop = cv2.resize(crop, (CROP_SIZE, CROP_SIZE)) if crop.size > 0 else np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)

            tracks.setdefault(track_id, {"crop": [], "box": [], "pose": []})
            tracks[track_id]["crop"].append(crop)
            tracks[track_id]["box"].append(box)
            tracks[track_id]["pose"].append(kpts)

            if len(tracks[track_id]["crop"]) > MAX_FRAMES_PER_TRACK:
                tracks[track_id]["crop"].pop(0)
                tracks[track_id]["box"].pop(0)
                tracks[track_id]["pose"].pop(0)

    frame_idx += 1
    if frame_idx % 50 == 0:
        print(f"진행: {frame_idx}프레임 처리, 현재 트랙 수: {len(tracks)}")

cap.release()

print(f"\n검출된 트랙 수: {len(tracks)}")
for tid, d in tracks.items():
    print(f"  track {tid}: {len(d['crop'])}프레임")

if not tracks:
    print("검출된 트랙이 없습니다.")
else:
    longest_id = max(tracks, key=lambda k: len(tracks[k]["crop"]))
    d = tracks[longest_id]

    np.savez(
        OUTPUT_NPZ,
        crop=np.array(d["crop"], dtype=np.uint8),
        box=np.array(d["box"], dtype=np.float32),
        pose=np.array(d["pose"], dtype=np.float32),
    )
    print(f"\n저장 완료: {OUTPUT_NPZ} (track {longest_id}, {len(d['crop'])}프레임)")
