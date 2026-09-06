"""
영상 첫 프레임에 grid를 그려서 저장 -> 사용자가 좌표를 보고 직접 입력
사용법: uv run python src/mark_crosswalk.py data/test_clip2.mp4
"""
import cv2
import json
import sys

VIDEO_PATH = sys.argv[1] if len(sys.argv) > 1 else "data/test_clip2.mp4"
OUTPUT_IMG = "crosswalk_grid.jpg"
OUTPUT_JSON = "crosswalk_zone.json"
STEP = 100

cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
cap.release()

if not ret:
    print("영상을 읽을 수 없습니다.")
    sys.exit(1)

h, w = frame.shape[:2]
for x in range(0, w, STEP):
    cv2.line(frame, (x, 0), (x, h), (0, 0, 255), 1)
    cv2.putText(frame, str(x), (x + 2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
for y in range(0, h, STEP):
    cv2.line(frame, (0, y), (w, y), (0, 0, 255), 1)
    cv2.putText(frame, str(y), (2, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

cv2.imwrite(OUTPUT_IMG, frame)
print(f"grid 이미지 저장됨: {OUTPUT_IMG}")
print(f"영상 크기: {w} x {h}")
print("\nVS Code에서 이 이미지를 열어서 횡단보도 네 모서리 좌표를 확인하세요.")
print("확인한 후 아래에 순서대로 입력하세요 (형식: x,y)\n")

points = []
labels = ["좌상단", "우상단", "우하단", "좌하단"]
for label in labels:
    while True:
        raw = input(f"{label} 좌표 입력 (예: 660,570): ").strip()
        try:
            x_str, y_str = raw.split(",")
            points.append([int(x_str.strip()), int(y_str.strip())])
            break
        except ValueError:
            print("형식이 잘못됐습니다. 'x,y' 형식으로 다시 입력하세요.")

with open(OUTPUT_JSON, "w") as f:
    json.dump({"video": VIDEO_PATH, "points": points}, f, indent=2)

print(f"\n저장 완료: {OUTPUT_JSON}")
print(points)
