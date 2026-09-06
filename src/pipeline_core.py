"""
사람+차량 통합 추적 + 횡단보도 폴리곤 판정 + 프레임별 박스 저장
"""
import math
import numpy as np
import cv2
from ultralytics import YOLO

CLASS_NAMES = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

def _order_polygon_points(points):
    """4점을 시계방향으로 정렬.

    사용자가 클릭 순서를 틀리면 폴리곤이 X자로 꼬여
    pointPolygonTest와 면적 계산이 모두 왜곡되므로 반드시 정렬한다.
    무게중심 기준 각도로 정렬하면 어떤 순서로 찍어도 볼록 사각형이 된다.
    """
    pts = np.array(points, dtype=np.float32)
    cx = float(np.mean(pts[:, 0]))
    cy = float(np.mean(pts[:, 1]))

    def angle(p):
        return math.atan2(p[1] - cy, p[0] - cx)

    ordered = sorted(pts.tolist(), key=angle)

    # 좌상단에서 시작하도록 회전 (x+y가 가장 작은 점)
    start = min(range(4), key=lambda i: ordered[i][0] + ordered[i][1])
    ordered = ordered[start:] + ordered[:start]

    return [[int(p[0]), int(p[1])] for p in ordered]

def _iou_overlap_ratio(box_a, box_b):
    """box_a가 box_b와 겹치는 비율 (교집합 면적 / box_a 면적)"""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(1e-6, (ax2 - ax1) * (ay2 - ay1))
    return inter / area_a


def _is_motorcycle_rider(d, frame_detections, overlap_threshold=0.5, frame_ratio_threshold=0.5):
    """이 person 트랙이 오토바이 탑승자일 가능성이 높은지 판정."""
    total = len(d["frames"])
    if total == 0:
        return False
    overlap_count = 0
    for frame_idx, person_box in zip(d["frames"], d["box"]):
        for det in frame_detections.get(frame_idx, []):
            if det["cls_id"] == 3 and _iou_overlap_ratio(person_box, det["box"]) > overlap_threshold:
                overlap_count += 1
                break
    return (overlap_count / total) > frame_ratio_threshold

def _box_polygon_overlap_ratio(box, polygon):
    """박스와 폴리곤의 겹침 면적 비율 (교집합 / 박스 면적).
    중심점 하나가 아니라 실제 면적으로 판단해야 차량이 부분 침범한 경우도 잡힌다."""
    x1, y1, x2, y2 = box
    box_area = max(1e-6, (x2 - x1) * (y2 - y1))

    px, py = polygon[:, 0], polygon[:, 1]
    min_x = int(min(x1, px.min())) - 1
    min_y = int(min(y1, py.min())) - 1
    max_x = int(max(x2, px.max())) + 1
    max_y = int(max(y2, py.max())) + 1
    w, h = max_x - min_x, max_y - min_y
    if w <= 0 or h <= 0:
        return 0.0

    box_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(box_mask,
                  (int(x1) - min_x, int(y1) - min_y),
                  (int(x2) - min_x, int(y2) - min_y), 1, -1)

    poly_mask = np.zeros((h, w), dtype=np.uint8)
    shifted = polygon.copy()
    shifted[:, 0] -= min_x
    shifted[:, 1] -= min_y
    cv2.fillPoly(poly_mask, [shifted], 1)

    inter = int(np.count_nonzero(box_mask & poly_mask))
    return inter / box_area

    inter = int(np.count_nonzero(box_mask & poly_mask))
    return inter / box_area


def _iou(box_a, box_b):
    """두 박스의 IoU (교집합 / 합집합)"""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    return inter / union if union > 1e-6 else 0.0


def _match_pose_to_box(person_box, pose_result, iou_threshold=0.3):
    """detect가 잡은 person_box에 해당하는 pose keypoints를 찾아 반환.

    기존 버그: pose_result.keypoints.xy[0]으로 무조건 첫 번째 사람의 자세를
    가져와서, 다중 보행자일 때 엉뚱한 사람의 자세가 매칭됐음.
    """
    empty = ([0.0] * 34, False)

    if pose_result.boxes is None or pose_result.keypoints is None:
        return empty
    if len(pose_result.boxes) == 0:
        return empty

    best_iou = 0.0
    best_idx = -1
    for i in range(len(pose_result.boxes)):
        pbox = pose_result.boxes.xyxy[i].tolist()
        iou = _iou(person_box, pbox)
        if iou > best_iou:
            best_iou = iou
            best_idx = i

    if best_idx < 0 or best_iou < iou_threshold:
        return empty

    kpts = pose_result.keypoints.xy[best_idx].flatten().tolist()
    if len(kpts) != 34:
        return empty

    return kpts, True

def _body_angle_to_crossing(kpts, crossing_dir):
    """어깨/골반 축과 횡단 진행방향 사이의 각도(0~90도).

    건너려는 사람은 몸통 축(어깨선)이 횡단 방향과 수직에 가깝고,
    옆으로 서 있거나 딴짓하는 사람은 평행에 가깝다.
    앞뒤 구분은 부감 각도에서 불안정하므로 하지 않는다.
    """

    axes = []
    # 어깨: 5(좌), 6(우) / 골반: 11(좌), 12(우)
    for i, j in [(5, 6), (11, 12)]:
        x1, y1 = kpts[i * 2], kpts[i * 2 + 1]
        x2, y2 = kpts[j * 2], kpts[j * 2 + 1]
        if x1 == 0 and y1 == 0:
            continue
        if x2 == 0 and y2 == 0:
            continue
        vx, vy = x2 - x1, y2 - y1
        if (vx * vx + vy * vy) ** 0.5 < 1e-6:
            continue
        axes.append((vx, vy))

    if not axes:
        return None

    vx = sum(a[0] for a in axes) / len(axes)
    vy = sum(a[1] for a in axes) / len(axes)
    norm = (vx * vx + vy * vy) ** 0.5
    if norm < 1e-6:
        return None
    vx, vy = vx / norm, vy / norm

    cx, cy = crossing_dir
    # 몸통 정면 = 어깨선의 법선
    fx, fy = -vy, vx
    dot = abs(fx * cx + fy * cy)
    dot = min(1.0, max(0.0, dot))
    return math.degrees(math.acos(dot))


def _crossing_direction(polygon):
    """횡단보도의 긴 축 = 보행자 진행 방향 (단위 벡터)"""
    best_len = -1
    best_vec = (1.0, 0.0)
    for i in range(len(polygon)):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % len(polygon)]
        vx, vy = float(p2[0] - p1[0]), float(p2[1] - p1[1])
        length = (vx * vx + vy * vy) ** 0.5
        if length > best_len:
            best_len = length
            best_vec = (vx / length, vy / length) if length > 1e-6 else (1.0, 0.0)
    return best_vec


def analyze_person_states(d, polygon, fps,
                          dist_threshold=1.5,
                          approach_dist_threshold=0.6,     # 추가
                          stop_speed_ratio=0.008,
                          window_frames=15,
                          dist_change_threshold=0.1,
                          body_angle_threshold=45.0,
                          move_angle_threshold=60.0,
                          max_dwell_sec=20.0):
    """보행자의 프레임별 상태를 판정.

    상태:
      crossing        - 발이 폴리곤 안 (실제 횡단 중)
      waiting_facing  - 근처 정지 + 몸이 횡단방향을 향함 (강한 대기 신호)
      waiting_idle    - 근처 정지 + 몸이 다른 방향 (딴짓 가능성)
      approaching     - 근처에서 횡단보도 쪽으로 접근 중
      leaving         - 멀어지는 중 (판정 제외)
      near_idle       - 근처지만 움직임 애매 (판정 제외)
      far             - 멀리 있음 (판정 제외)

    거리는 보행자 박스 높이로 정규화하여 원근을 보정한다.
    (박스 높이가 그 위치에서의 사람 키(약 1.7m)에 해당)
    """
    n = len(d["frames"])
    crossing_dir = _crossing_direction(polygon)

    # 프레임별 기본 지표 계산
    foot = []       # 발 위치 (박스 하단 중앙)
    heights = []    # 박스 높이
    norm_dist = []  # 정규화 거리 (음수면 폴리곤 안)

    for i in range(n):
        x1, y1, x2, y2 = d["box"][i]
        fx, fy = (x1 + x2) / 2.0, y2
        h = max(1e-6, y2 - y1)
        foot.append((fx, fy))
        heights.append(h)
        signed = cv2.pointPolygonTest(polygon, (float(fx), float(fy)), True)
        norm_dist.append(-signed / h)   # 안이면 음수, 밖이면 양수

    # 프레임별 정규화 이동속도
    norm_speed = [0.0] * n
    for i in range(1, n):
        dx = foot[i][0] - foot[i - 1][0]
        dy = foot[i][1] - foot[i - 1][1]
        norm_speed[i] = ((dx * dx + dy * dy) ** 0.5) / heights[i]

    states = []
    dwell_run = 0

    for i in range(n):
        nd = norm_dist[i]

        # 1) 폴리곤 안 = 실제 횡단 중
        if nd <= 0:
            states.append("crossing")
            dwell_run = 0
            continue

        # 2) 거리 조건 미달
        if nd > dist_threshold:
            states.append("far")
            dwell_run = 0
            continue

        # 3) 정지 상태 판별 (최근 window_frames 연속 저속)
        lo = max(0, i - window_frames + 1)
        recent = norm_speed[lo:i + 1]
        is_stopped = len(recent) >= window_frames and all(s < stop_speed_ratio for s in recent)

        if is_stopped:
            dwell_run += 1
            if dwell_run / fps > max_dwell_sec:
                states.append("near_idle")   # 장기 체류 → 대기로 보지 않음
                continue

            angle = None
            if d["pose_ok"][i]:
                angle = _body_angle_to_crossing(d["pose"][i], crossing_dir)

            if angle is not None and angle <= body_angle_threshold:
                states.append("waiting_facing")
            else:
                states.append("waiting_idle")
            continue

        dwell_run = 0

        # 4) 접근/이탈 추세 + 이동 방향
        if i >= window_frames:
            delta = norm_dist[i - window_frames] - nd

            # 이동 방향이 횡단 방향과 나란한지 확인.
            # 인도를 따라 걷는 사람은 횡단보도와 직각으로 움직이므로
            # 거리가 줄어들어도 '건너려는 접근'이 아니다.
            mdx = foot[i][0] - foot[i - window_frames][0]
            mdy = foot[i][1] - foot[i - window_frames][1]
            mlen = (mdx * mdx + mdy * mdy) ** 0.5

            move_aligned = False
            if mlen > 1e-6:
                mdx, mdy = mdx / mlen, mdy / mlen
                dot = abs(mdx * crossing_dir[0] + mdy * crossing_dir[1])
                dot = min(1.0, max(0.0, dot))
                move_angle = math.degrees(math.acos(dot))
                move_aligned = move_angle <= move_angle_threshold

            if delta > dist_change_threshold and nd <= approach_dist_threshold:
                states.append("approaching")
                continue
            if delta < -dist_change_threshold:
                states.append("leaving")
                continue

        states.append("near_idle")

    return {
        "states": states,
        "norm_dist": [round(v, 3) for v in norm_dist],
        "norm_speed": [round(v, 4) for v in norm_speed],
    }


def _analyze_vehicle_motion(points, fps,
                            stop_ratio_threshold=0.002,
                            creep_ratio_threshold=0.012,
                            min_stop_duration_sec=0.35):
    """차량의 운동 상태를 3단계로 판정.

    도로교통법 제27조의 '일시정지'는 완전 정지를 의미하므로,
    서행(creeping)과 정지(stopped)를 반드시 구분해야 한다.

    기존 min_speed 방식의 문제: 구간 내 최솟값 하나만 보므로
    추적 흔들림 한 프레임에 정지로 오판됨.
    """
    if len(points) < 2:
        return {"motion_state": "moving", "stopped": False,
                "avg_speed": None, "min_speed": None,
                "min_norm_speed": None, "max_stop_duration_sec": 0.0,
                "reason": "insufficient_frames"}

    norm_speeds = []
    raw_speeds = []
    for i in range(1, len(points)):
        dx = points[i]["ax"] - points[i - 1]["ax"]
        dy = points[i]["ay"] - points[i - 1]["ay"]
        dist = (dx ** 2 + dy ** 2) ** 0.5
        raw_speeds.append(dist)

        diag = points[i]["diag"]
        norm_speeds.append(dist / diag if diag > 1e-6 else 0.0)

    max_run = 0
    cur_run = 0
    for s in norm_speeds:
        if s < stop_ratio_threshold:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 0

    max_stop_duration_sec = max_run / fps if fps > 0 else 0.0

    if max_stop_duration_sec >= min_stop_duration_sec:
        motion_state = "stopped"
        reason = "sustained_full_stop"
    else:
        avg_norm = sum(norm_speeds) / len(norm_speeds)
        if avg_norm < creep_ratio_threshold:
            motion_state = "creeping"
            reason = "slow_but_never_fully_stopped"
        else:
            motion_state = "moving"
            reason = "passed_at_normal_speed"

    return {
        "motion_state": motion_state,
        "stopped": motion_state == "stopped",
        "avg_speed": round(sum(raw_speeds) / len(raw_speeds), 2),
        "min_speed": round(min(raw_speeds), 2),
        "min_norm_speed": round(min(norm_speeds), 5),
        "max_stop_duration_sec": round(max_stop_duration_sec, 2),
        "reason": reason,
    }


def get_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("영상을 읽을 수 없습니다.")
    return frame


def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"fps": fps, "width": w, "height": h}


def run_pipeline(video_path, crosswalk_points,
                  conf=0.35, stop_speed_threshold=5.0):
    crosswalk_points = _order_polygon_points(crosswalk_points)
    polygon = np.array(crosswalk_points, dtype=np.int32)

    yolo = YOLO("yolo11m.pt")
    pose_model = YOLO("yolo11m-pose.pt")
    cap = cv2.VideoCapture(video_path)


    person_tracks = {}
    vehicle_tracks = {}
    frame_detections = {}
    frame_idx = 0

    def in_zone(cx, cy):
        return cv2.pointPolygonTest(polygon, (float(cx), float(cy)), False) >= 0

    # 프레임을 직접 읽어서 YOLO에 넘긴다.
    # 기존에는 yolo.track(source=video_path)가 내부적으로 영상을 읽고
    # cap.read()로 또 따로 읽어서 두 스트림이 어긋날 수 있었다.
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        det_r = yolo.track(
            frame, classes=[0, 2, 3, 5, 7], persist=True,
            device=0, conf=conf, tracker="my_bytetrack.yaml", verbose=False,
            imgsz=1920
        )[0]

        frame_detections[frame_idx] = []

        if det_r.boxes is not None and det_r.boxes.id is not None:
            pose_r = pose_model.predict(frame, device=0, verbose=False, imgsz=1920)[0]

            for i, track_id in enumerate(det_r.boxes.id.tolist()):
                cls_id = int(det_r.boxes.cls[i].item())
                box = det_r.boxes.xyxy[i].tolist()
                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                frame_detections[frame_idx].append({"track_id": int(track_id), "cls_id": cls_id, "box": box})

                if cls_id == 0:
                    # detect 박스와 IoU가 가장 큰 pose를 매칭 (첫 번째 사람 고정 버그 수정)
                    kpts, pose_matched = _match_pose_to_box(box, pose_r)
                    person_tracks.setdefault(track_id, {
                        "box": [], "pose": [], "pose_ok": [],
                        "cx": [], "cy": [], "frames": []
                    })

                    # 로직 2 판정용 데이터는 전체 궤적 보관
                    # (거리 변화 추세, 대기 시간 계산에 전체 이력이 필요함)
                    person_tracks[track_id]["box"].append(box)
                    person_tracks[track_id]["pose"].append(kpts)
                    person_tracks[track_id]["pose_ok"].append(pose_matched)
                    person_tracks[track_id]["cx"].append(cx)
                    person_tracks[track_id]["cy"].append(cy)
                    person_tracks[track_id]["frames"].append(frame_idx)
                else:
                    # ax, ay = 차량 접지점(박스 하단 중앙). 중심점보다 궤적이 안정적이고
                    # 도로 위 실제 위치에 가까워 원근 왜곡이 덜하다.
                    ax, ay = (x1 + x2) / 2, y2
                    diag = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                    overlap = _box_polygon_overlap_ratio(box, polygon)
                    vehicle_tracks.setdefault(track_id, []).append({
                        "frame": frame_idx, "cx": cx, "cy": cy,
                        "ax": ax, "ay": ay, "diag": diag,
                        "overlap": overlap, "box": box,
                    })

        frame_idx += 1

    cap.release()

    def near_zone_score(d):
        """폴리곤 근처(정규화 거리 기준)에 온 적이 있는가.

        폴리곤 '안'에 들어온 적만 따지면, 차량 때문에 건너지 못한 보행자가
        후보에서 통째로 제외되어 위반을 놓치게 된다.
        """
        count = 0
        for i in range(len(d["box"])):
            x1, y1, x2, y2 = d["box"][i]
            fx, fy = (x1 + x2) / 2.0, y2          # 발 위치
            h = max(1e-6, y2 - y1)                 # 박스 높이로 정규화
            signed = cv2.pointPolygonTest(polygon, (float(fx), float(fy)), True)
            if (-signed / h) <= 2.0:               # 거리 판정 임계값(1.5)보다 여유 있게
                count += 1
        return count

    MAX_PERSON_CANDIDATES = 15  # 극단적으로 혼잡한 교차로 대비 안전장치일 뿐, 일반 상황에선 걸릴 일 없음

    persons_result = []
    if person_tracks:
        candidates = [
            (pid, d) for pid, d in person_tracks.items()
            if not _is_motorcycle_rider(d, frame_detections)
        ]

        # 핵심 필터: 내가 지정한 횡단보도 폴리곤에 실제로 들어온 적 있는 사람만 후보로 삼음
        near_candidates = [(pid, d) for pid, d in candidates if near_zone_score(d) > 0]

        if len(near_candidates) > MAX_PERSON_CANDIDATES:
            # 정상적인 영상에서는 거의 발생하지 않음. 발생하면 성능 문제이지 판정 로직 문제는 아님
             near_candidates.sort(key=lambda kv: len(kv[1]["frames"]), reverse=True)
             near_candidates = near_candidates[:MAX_PERSON_CANDIDATES]

        # 근처에 온 사람이 아예 없으면 (오검출 등으로) 필터링 이전 전체로 폴백
        if not near_candidates:
            near_candidates = candidates if candidates else list(person_tracks.items())

        for pid, d in near_candidates:
            # 로직 2: 프레임별 보행자 상태 판정
            fps_val = get_video_info(video_path)["fps"]
            analysis = analyze_person_states(d, polygon, fps_val)
            states = analysis["states"]

            # 판정 대상이 되는 상태만 추출
            JUDGE_STATES = {"crossing", "waiting_facing", "waiting_idle", "approaching"}
            judge_frames = [
                d["frames"][i] for i, s in enumerate(states) if s in JUDGE_STATES
            ]

            # 상태별 프레임 수 집계 (근거 표시용)
            state_counts = {}
            for s in states:
                state_counts[s] = state_counts.get(s, 0) + 1

            persons_result.append({
                "track_id": pid,
                "frames": len(d["frames"]),
                "frame_range": [d["frames"][0], d["frames"][-1]],
                "judge_frames": judge_frames,
                "judge_range": [judge_frames[0], judge_frames[-1]] if judge_frames else None,
                "state_counts": state_counts,
                "min_norm_dist": min(analysis["norm_dist"]) if analysis["norm_dist"] else None,
                # 거리 변화 추이 확인용 (10프레임 간격 샘플링)
                "norm_dist_sample": analysis["norm_dist"][::10],
                "frame_states": {d["frames"][i]: states[i] for i in range(len(states))},
            })

    fps = get_video_info(video_path)["fps"]
    VEHICLE_OVERLAP_THRESHOLD = 0.02  # 박스 면적의 2% 이상 겹치면 횡단보도 침범으로 간주

    vehicle_results = []
    for vid, points in vehicle_tracks.items():
        # 중심점이 아니라 박스-폴리곤 겹침 면적으로 판단 → 부분 침범도 감지
        in_zone_points = [p for p in points if p["overlap"] > VEHICLE_OVERLAP_THRESHOLD]

        if len(in_zone_points) == 0:
            continue

        motion = _analyze_vehicle_motion(in_zone_points, fps)

        # 폴리곤 안에 있던 프레임 집합 (보행자 판정구간과 교차시키기 위함)
        zone_frames = set(p["frame"] for p in in_zone_points)

        vehicle_results.append({
            "track_id": vid,
            "frame_range": [in_zone_points[0]["frame"], in_zone_points[-1]["frame"]],
            "frames_in_zone": len(in_zone_points),
            "zone_frames": zone_frames,
            "max_overlap": round(max(p["overlap"] for p in in_zone_points), 3),
            **motion,
        })

    return {
        "persons": persons_result,
        "vehicles": vehicle_results,
        "frame_detections": frame_detections,
        "crosswalk_points": crosswalk_points,
    }


def determine_violation(result):
    """프레임 단위로 위반 여부를 판정.

    특정 보행자와 특정 차량을 짝지어 비교하지 않는다.
    추적이 끊겨 트랙이 조각나도 "그 프레임에 대기 중인 보행자가 있었고,
    그때 차량이 정지 없이 횡단보도를 통과했다"는 사실은 보존되기 때문이다.

    반환: (is_violation, violation_frames, violation_events)
    """
    JUDGE_STATES = {"crossing", "waiting_facing", "waiting_idle", "approaching"}
    STRONG_STATES = {"crossing", "waiting_facing"}

    # 1) 프레임별 보행자 상태 집계
    ped_frames = {}          # frame -> set(상태)
    ped_tracks_at = {}       # frame -> set(track_id)  (근거 표시용)
    for person in result["persons"]:
        pid = person["track_id"]
        for f, state in person.get("frame_states", {}).items():
            if state in JUDGE_STATES:
                ped_frames.setdefault(f, set()).add(state)
                ped_tracks_at.setdefault(f, set()).add(pid)

    if not ped_frames:
        return False, set(), []

    # 2) 프레임별 차량 상태 집계 (정지하지 않은 차만)
    veh_frames = {}          # frame -> list of (track_id, motion_state)
    for v in result["vehicles"]:
        motion_state = v.get("motion_state", "moving")
        if motion_state == "stopped":
            continue          # 완전 정지 = 의무 이행
        for f in v.get("zone_frames", set()):
            veh_frames.setdefault(f, []).append((v["track_id"], motion_state))

    # 3) 교집합 = 위반 프레임
    violation_frames = set(ped_frames.keys()) & set(veh_frames.keys())
    if not violation_frames:
        return False, set(), []

    # 4) 차량 단위로 이벤트 생성
    #    같은 차량이 횡단보도를 한 번 통과하는 동안 발생한 위반은
    #    검출이 중간에 끊기더라도 하나의 사건으로 취급한다.
    events = []
    by_vehicle = {}

    for f in sorted(violation_frames):
        for vid, ms in veh_frames[f]:
            by_vehicle.setdefault(vid, {"frames": [], "motions": set()})
            by_vehicle[vid]["frames"].append(f)
            by_vehicle[vid]["motions"].add(ms)

    for vid, info in by_vehicle.items():
        frames = sorted(set(info["frames"]))
        start, end = frames[0], frames[-1]

        states = set()
        pids = set()
        for f in frames:
            states |= ped_frames[f]
            pids |= ped_tracks_at[f]

        has_strong = bool(states & STRONG_STATES)
        motions = info["motions"]
        if "moving" in motions and has_strong:
            severity = "high"
        elif has_strong:
            severity = "medium"
        else:
            severity = "review_needed"
            
        events.append({
            "frame_range": [start, end],
            "frame_count": len(frames),
            "person_states": sorted(states),
            "person_track_ids": sorted(pids),
            "vehicle_track_ids": [vid],
            "vehicle_motion_states": sorted(motions),
            "severity": severity,
        })

    events.sort(key=lambda e: e["frame_range"][0])

    return True, violation_frames, events


def capture_violation_snapshots(video_path, result, violation_events,
                                 output_dir="data/snapshots"):
    """각 위반 이벤트의 대표 프레임을 이미지로 저장.

    crossing 상태가 포함된 구간이 가장 명확한 증거이므로 우선 선택하고,
    없으면 겹침 구간의 중간 프레임을 사용한다.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    fps = get_video_info(video_path)["fps"]
    polygon = np.array(result["crosswalk_points"], dtype=np.int32)
    frame_detections = result["frame_detections"]

    # 이벤트별 대표 프레임 번호 결정
    # 이벤트마다 두 시점을 뽑는다.
    #   enter: 차량이 횡단보도에 진입하는 순간 (보행자가 이미 대기/횡단 중이었다는 증거)
    #   pass : 차량이 통과를 마치는 순간 (정지하지 않고 지나갔다는 증거)
    targets = []
    for idx, ev in enumerate(violation_events):
        start, end = ev["frame_range"]

        enter_f, pass_f = start, end
        for v in result["vehicles"]:
            if v["track_id"] not in ev.get("vehicle_track_ids", []):
                continue
            zone = sorted(f for f in v.get("zone_frames", set()) if start <= f <= end)
            if zone:
                enter_f, pass_f = zone[0], zone[-1]
                break

        targets.append((idx, "enter", enter_f, ev))
        if pass_f != enter_f:
            targets.append((idx, "pass", pass_f, ev))

    if not targets:
        return []

    wanted = {f for _, _, f, _ in targets}
    frames = {}

    cap = cv2.VideoCapture(video_path)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in wanted:
            frames[idx] = frame.copy()
            if len(frames) == len(wanted):
                break
        idx += 1
    cap.release()

    saved = []
    for ev_idx, kind, fnum, ev in targets:
        frame = frames.get(fnum)
        if frame is None:
            continue

        img = frame.copy()
        h, w = img.shape[:2]

        cv2.polylines(img, [polygon], isClosed=True, color=(0, 255, 255), thickness=3)

        ped_ids = set(ev.get("person_track_ids", []))
        veh_ids = set(ev.get("vehicle_track_ids", []))

        for det in frame_detections.get(fnum, []):
            tid = det["track_id"]
            x1, y1, x2, y2 = map(int, det["box"])
            if det["cls_id"] == 0 and tid in ped_ids:
                color, label = (0, 255, 0), f"pedestrian {tid}"
            elif det["cls_id"] != 0 and tid in veh_ids:
                color, label = (0, 0, 255), f"VIOLATING VEHICLE {tid}"
            else:
                continue
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            cv2.putText(img, label, (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        snap_sec = round(fnum / fps, 1) if fps else "-"
        kind_label = "vehicle entering crosswalk" if kind == "enter" else "vehicle passed through"
        header = f"{snap_sec}s  {kind_label}  |  pedestrian: {', '.join(ev.get('person_states', []))}"
        cv2.rectangle(img, (0, 0), (w, 45), (0, 0, 0), -1)
        cv2.putText(img, header, (15, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        path = os.path.join(output_dir, f"violation_{ev_idx}_{kind}.jpg")
        cv2.imwrite(path, img)
        saved.append(path)
        ev.setdefault("snapshots", {})[kind] = path
        ev.setdefault("snapshot_sec", {})[kind] = snap_sec
        if kind == "enter":
            ev["snapshot_path"] = path      # 기존 호환용

    return saved

def render_annotated_video(video_path, result, is_violation, output_path="data/annotated_output.mp4"):
    info = get_video_info(video_path)
    fps, w, h = info["fps"], info["width"], info["height"]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    tmp_path = output_path.replace(".mp4", "_raw.mp4")
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (w, h))

    polygon = np.array(result["crosswalk_points"], dtype=np.int32)
    frame_detections = result["frame_detections"]

    _, violation_frames, _ = determine_violation(result)

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.polylines(frame, [polygon], isClosed=True, color=(0, 255, 255), thickness=2)

        for det in frame_detections.get(frame_idx, []):
            x1, y1, x2, y2 = map(int, det["box"])
            if det["cls_id"] == 0:
                color = (0, 255, 0)
                label = f"person {det['track_id']}"
            else:
                color = (255, 128, 0)
                label = f"vehicle {det['track_id']}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if frame_idx in violation_frames:
            cv2.putText(frame, "VIOLATION CANDIDATE", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    return tmp_path


def reencode_for_browser(input_path, output_path="data/annotated_final.mp4"):
    import subprocess
    subprocess.run([
        "/usr/bin/ffmpeg", "-y", "-i", input_path,
        "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        output_path
    ], check=True, capture_output=True)
    return output_path
