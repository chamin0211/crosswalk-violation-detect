import cv2
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
from PIL import Image, ImageDraw

from src.pipeline_core import (get_first_frame, run_pipeline, render_annotated_video,
                               reencode_for_browser, determine_violation, get_video_info,
                               capture_violation_snapshots, _order_polygon_points)

st.set_page_config(page_title="비신호 횡단보도 위반 감지", layout="wide")

VIDEO_PATH = "data/uploaded_video.mp4"

MOTION_LABEL = {
    "stopped": "정지함 (의무 이행)",
    "creeping": "서행 (정지 아님)",
    "moving": "정지하지 않고 통과",
}

STATE_LABEL = {
    "crossing": "횡단 중",
    "waiting_facing": "대기 (도로 방향)",
    "waiting_idle": "대기 (기타)",
    "approaching": "횡단보도로 접근 중",
    "leaving": "멀어지는 중",
    "near_idle": "근처 체류",
    "far": "횡단보도와 무관",
}

SEVERITY_LABEL = {
    "high": "위반 가능성 높음",
    "medium": "검토 필요",
    "review_needed": "검토 필요",
}


def build_event_summary(ev):
    """검토자가 읽을 수 있는 서술형 요약 생성."""
    s, e = ev["시각(초)"]
    states = [STATE_LABEL.get(x, x) for x in ev["person_states"]]
    vids = ", ".join(str(v) for v in ev["vehicle_track_ids"])
    motions = [MOTION_LABEL.get(m, m) for m in ev["vehicle_motion_states"]]
    return (
        f"{s}초 ~ {e}초 구간에서 보행자가 '{' / '.join(states)}' 상태였으나, "
        f"차량(ID {vids})이 {' / '.join(motions)} 하였습니다. "
        f"도로교통법 제27조에 따른 일시정지 의무 위반 후보로 판단됩니다."
    )


def reset_to_upload():
    for k in ["analysis", "final_video"]:
        st.session_state.pop(k, None)
    st.session_state.points = []
    st.session_state.page = "upload"


if "points" not in st.session_state:
    st.session_state.points = []
if "page" not in st.session_state:
    st.session_state.page = "upload"


# ============================================================
# 1. 업로드 + 횡단보도 지정
# ============================================================
if st.session_state.page == "upload":
    st.title("비신호 횡단보도 보행자 보호의무 위반 감지")
    st.caption("본 결과는 참고용이며, 최종 판단은 담당자가 합니다.")

    uploaded_file = st.file_uploader(
        "영상 업로드", type=["mp4", "mov", "avi"],
        help="한 번에 하나의 영상만 분석합니다."
    )

    has_video = False
    if uploaded_file is not None:
        with open(VIDEO_PATH, "wb") as f:
            f.write(uploaded_file.read())
        has_video = True

    if has_video:
        frame = get_first_frame(VIDEO_PATH)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = frame_rgb.shape[:2]

        display_w = 820
        scale = display_w / orig_w
        display_h = int(orig_h * scale)
        img_display = Image.fromarray(frame_rgb).resize((display_w, display_h))

        # 이미 찍은 점과 연결선을 그려서, 사용자가 어디를 클릭했는지
        # 그리고 폴리곤이 꼬이지 않았는지 바로 확인할 수 있게 한다.
        if st.session_state.points:
            draw = ImageDraw.Draw(img_display)
            # 4점이 다 찍히면 실제 판정에 쓰이는 순서(자동 정렬)로 미리보기
            preview_pts = (_order_polygon_points(st.session_state.points)
                           if len(st.session_state.points) == 4
                           else st.session_state.points)
            disp_pts = [(int(x * scale), int(y * scale)) for x, y in preview_pts]

            if len(disp_pts) >= 2:
                line_pts = disp_pts + ([disp_pts[0]] if len(disp_pts) == 4 else [])
                draw.line(line_pts, fill=(255, 220, 0), width=3)

            for i, (px, py) in enumerate(disp_pts, 1):
                r = 7
                draw.ellipse([px - r, py - r, px + r, py + r],
                             fill=(255, 60, 60), outline=(255, 255, 255), width=2)
                draw.text((px + 11, py - 7), str(i), fill=(255, 255, 255))

        # wide 레이아웃에서 콘텐츠가 양끝으로 벌어지지 않도록 가운데로 모은다
        _, main, _ = st.columns([1, 5, 1])

        with main:
            col_img, col_ctrl = st.columns([3, 2], gap="medium")

            with col_img:
                st.write("횡단보도 네 모서리를 클릭하세요 (순서 무관)")
                coords = streamlit_image_coordinates(img_display, key="click")

            if coords is not None:
                orig_x = int(coords["x"] / scale)
                orig_y = int(coords["y"] / scale)
                pt = [orig_x, orig_y]
                if len(st.session_state.points) < 4 and (
                    not st.session_state.points or pt != st.session_state.points[-1]
                ):
                    st.session_state.points.append(pt)
                    st.rerun()

            with col_ctrl:
                st.markdown("##### 횡단보도 지정")
                st.progress(len(st.session_state.points) / 4,
                            text=f"{len(st.session_state.points)} / 4 지정됨")

                if st.session_state.points:
                    st.caption(" · ".join(f"({x}, {y})" for x, y in st.session_state.points))

                if st.button("좌표 초기화", use_container_width=True):
                    st.session_state.points = []
                    st.rerun()

                if st.button("분석 시작", type="primary", use_container_width=True,
                             disabled=len(st.session_state.points) != 4):
                    st.session_state.page = "analyzing"
                    st.rerun()

                st.markdown("---")
                st.caption(
                    "판정 대상 횡단보도의 네 모서리를 클릭해 지정합니다. "
                    "지정한 구간을 통과하는 차량과, 그 구간으로 접근하거나 "
                    "건너는 보행자만 분석 대상이 됩니다."
                )


# ============================================================
# 2. 분석 진행
# ============================================================
elif st.session_state.page == "analyzing":
    st.title("분석 중")
    st.caption("영상 길이와 등장 인원에 따라 수 분이 걸릴 수 있습니다.")

    prog = st.progress(0, text="영상 준비 중...")

    prog.progress(10, text="사람과 차량을 추적하는 중...")
    result = run_pipeline(VIDEO_PATH, st.session_state.points)

    if not result["persons"]:
        prog.empty()
        st.error("보행자를 찾지 못했습니다. 다른 영상이나 다른 구간으로 시도해 보세요.")
        if st.button("돌아가기"):
            reset_to_upload()
            st.rerun()
        st.stop()

    prog.progress(85, text="위반 여부를 판정하는 중...")
    is_violation, violation_frames, violation_events = determine_violation(result)

    fps = get_video_info(VIDEO_PATH)["fps"]
    for ev in violation_events:
        s, e = ev["frame_range"]
        ev["시각(초)"] = [round(s / fps, 1), round(e / fps, 1)]

    st.session_state.analysis = {
        "result": result,
        "is_violation": is_violation,
        "events": violation_events,
        "video_path": VIDEO_PATH,
    }

    if is_violation:
        prog.progress(95, text="증거 화면을 저장하는 중...")
        capture_violation_snapshots(VIDEO_PATH, result, violation_events)

    prog.progress(100, text="완료")
    st.session_state.page = "result"
    st.rerun()


# ============================================================
# 3. 결과
# ============================================================
elif st.session_state.page == "result" and "analysis" in st.session_state:
    a = st.session_state.analysis
    events = a["events"]
    fps = get_video_info(a["video_path"])["fps"]

    hcol1, hcol2 = st.columns([5, 1])
    with hcol1:
        if a["is_violation"]:
            high = sum(1 for e in events if e["severity"] == "high")
            st.error(f"**위반 후보 {len(events)}건 감지** · 위반 가능성 높음 {high}건")
        else:
            st.success("**위반 후보가 감지되지 않았습니다**")
    with hcol2:
        if st.button("새 영상 분석", use_container_width=True):
            reset_to_upload()
            st.rerun()

    left, right = st.columns([1, 1], gap="medium")

    # ---- 왼쪽: 위반 순간 (한 건씩 선택해서 크게 보기) ----
    with left:
        st.markdown("##### 위반 순간")
        if not events:
            st.info("감지된 위반 후보가 없습니다.")
        else:
            labels = [
                f"{'🔴' if e['severity'] == 'high' else '🟡'} "
                f"{e['시각(초)'][0]}s ~ {e['시각(초)'][1]}s"
                for e in events
            ]
            idx = st.radio(
                "위반 후보 선택", range(len(events)),
                format_func=lambda i: labels[i],
                horizontal=True, label_visibility="collapsed",
            )
            ev = events[idx]
            snaps = ev.get("snapshots", {})
            secs = ev.get("snapshot_sec", {})

            if len(snaps) >= 2:
                kind = st.radio(
                    "시점 선택", ["enter", "pass"],
                    format_func=lambda k: (
                        f"횡단보도 진입 ({secs.get('enter', '-')}s)" if k == "enter"
                        else f"통과 완료 ({secs.get('pass', '-')}s)"
                    ),
                    horizontal=True, label_visibility="collapsed",
                    key=f"snapkind_{idx}",
                )
                st.image(snaps[kind], use_container_width=True)
            elif snaps:
                st.image(list(snaps.values())[0], use_container_width=True)
            elif ev.get("snapshot_path"):
                st.image(ev["snapshot_path"], use_container_width=True)

            st.caption(build_event_summary(ev))

    # ---- 오른쪽: 상세 정보 ----
    with right:
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["위반 상세", "차량 기록", "보행자 기록", "분석 영상", "개발자"]
        )

        with tab1:
            if not events:
                st.info("감지된 위반 후보가 없습니다.")
            for i, ev in enumerate(events, 1):
                st.markdown(f"**위반 후보 {i}**")
                st.dataframe({
                    "항목": ["발생 시각", "지속 시간", "보행자 상태", "차량 ID",
                             "차량 상태", "최대 정지 시간", "판정 등급"],
                    "내용": [
                        f"{ev['시각(초)'][0]}초 ~ {ev['시각(초)'][1]}초",
                        f"{ev['frame_count'] / fps:.1f}초",
                        " / ".join(STATE_LABEL.get(x, x) for x in ev["person_states"]),
                        ", ".join(str(v) for v in ev["vehicle_track_ids"]),
                        " / ".join(MOTION_LABEL.get(m, m) for m in ev["vehicle_motion_states"]),
                        next((f"{v.get('max_stop_duration_sec', 0)}초"
                              for v in a["result"]["vehicles"]
                              if v["track_id"] in ev["vehicle_track_ids"]), "-"),
                        SEVERITY_LABEL.get(ev["severity"], ev["severity"]),
                    ],
                }, hide_index=True, use_container_width=True)

        with tab2:
            rows = []
            for v in a["result"]["vehicles"]:
                fr = v["frame_range"]
                rows.append({
                    "차량 ID": v["track_id"],
                    "통과 시각": f"{fr[0]/fps:.1f}s ~ {fr[1]/fps:.1f}s",
                    "판정": MOTION_LABEL.get(v.get("motion_state"), "-"),
                    "최대 정지": f"{v.get('max_stop_duration_sec', 0)}초",
                    "침범 정도": f"{min(v.get('max_overlap', 0), 1.0)*100:.0f}%",
                })
            st.dataframe(rows, hide_index=True, use_container_width=True)
            st.caption("침범 정도: 차량 박스가 횡단보도와 겹친 최대 비율")

        with tab3:
            rows = []
            for p in a["result"]["persons"]:
                fr = p["frame_range"]
                counts = p["state_counts"]
                main_state = max(counts, key=counts.get) if counts else "-"
                jr = p.get("judge_range")
                rows.append({
                    "보행자 ID": p["track_id"],
                    "등장 시각": f"{fr[0]/fps:.1f}s ~ {fr[1]/fps:.1f}s",
                    "주요 상태": STATE_LABEL.get(main_state, main_state),
                    "판정 대상": f"{jr[0]/fps:.1f}s ~ {jr[1]/fps:.1f}s" if jr else "해당 없음",
                    "최근접 거리": f"{p['min_norm_dist']:.2f}" if p.get("min_norm_dist") is not None else "-",
                })
            st.dataframe(rows, hide_index=True, use_container_width=True)
            st.caption("최근접 거리: 보행자 키 대비 비율. 0 이하면 횡단보도 안에 있었음을 의미")

        with tab4:
            if st.button("영상 생성 (시간이 걸립니다)"):
                with st.spinner("영상 렌더링 중..."):
                    raw_path = render_annotated_video(
                        a["video_path"], a["result"], a["is_violation"]
                    )
                    st.session_state.final_video = reencode_for_browser(raw_path)
            if "final_video" in st.session_state:
                st.video(st.session_state.final_video)

        with tab5:
            st.write("차량 원본 데이터")
            st.write([{k: v for k, v in v_.items() if k != "zone_frames"}
                      for v_ in a["result"]["vehicles"]])
            st.write("보행자 원본 데이터")
            st.write([{k: v for k, v in p.items()
                       if k not in ("frame_states", "judge_frames")}
                      for p in a["result"]["persons"]])
            st.write("위반 이벤트 원본")
            st.write(events)


# 결과 페이지인데 분석 데이터가 없는 비정상 상태 (세션 초기화 등)
else:
    st.warning("분석 데이터가 없습니다.")
    if st.button("처음으로"):
        reset_to_upload()
        st.rerun()