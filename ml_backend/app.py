"""
VectraTrack ML — Streamlit Web Application (v3.0)
===================================================
New in v3.0:
  - ByteTrack 2-stage association (high/low confidence split)
  - Object Counting + Virtual Line Crossing (configurable line, IN/OUT direction)
  - Trajectory Heatmap overlay + PNG download
  - Enhanced Telemetry CSV (per-frame + crossing events as separate downloads)
"""

import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import os
import time
import pandas as pd
from custom_tracker import CustomSortTracker
from analytics import LineCrossingCounter, HeatmapAccumulator, TelemetryLogger

# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="VectraTrack ML", page_icon="🎯",
                   layout="wide", initial_sidebar_state="expanded")

# ── COCO Classes ─────────────────────────────────────────────────────────────
COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote",
    "keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book",
    "clock","vase","scissors","teddy bear","hair drier","toothbrush",
]

def get_class_colour(cls_id):
    palette = [(0,255,170),(255,0,200),(0,220,255),(255,180,0),
               (180,0,255),(0,255,80),(255,80,0),(0,100,255)]
    return palette[int(cls_id) % len(palette)]

# ── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&display=swap');
header{visibility:hidden;} footer{visibility:hidden;} .stDeployButton{display:none;}
.stApp{background:radial-gradient(ellipse at 20% 20%,#0d0221 0%,#05010a 60%,#010010 100%);
       font-family:'Share Tech Mono',monospace;color:#d0c0f0;}
[data-testid="stSidebar"]{background:rgba(10,5,25,0.9)!important;
    border-right:1px solid rgba(168,85,247,0.3);backdrop-filter:blur(12px);}
[data-testid="stSidebar"]*{color:#c4b5fd!important;font-family:'Share Tech Mono',monospace!important;}
h1{font-family:'Orbitron',sans-serif!important;color:#fff!important;
   text-shadow:0 0 5px #fff,0 0 15px #a855f7,0 0 40px #a855f7,0 0 80px #7c3aed;
   letter-spacing:4px;text-transform:uppercase;text-align:center;}
h2,h3{font-family:'Orbitron',sans-serif!important;color:#c084fc!important;
      letter-spacing:2px;text-shadow:0 0 10px rgba(168,85,247,0.5);}
[data-testid="stMetric"]{background:rgba(168,85,247,0.08);
    border:1px solid rgba(168,85,247,0.25);border-radius:12px;padding:1rem;
    box-shadow:0 0 15px rgba(168,85,247,0.1);}
[data-testid="stMetricLabel"]{color:#a78bfa!important;font-size:0.75rem!important;}
[data-testid="stMetricValue"]{color:#fff!important;font-family:'Orbitron',sans-serif!important;}
.stButton>button,.stDownloadButton>button{
    background:linear-gradient(90deg,rgba(124,58,237,0.15) 0%,rgba(168,85,247,0.35) 100%)!important;
    color:#fff!important;border:1px solid #a855f7!important;border-radius:8px!important;
    box-shadow:0 0 12px rgba(168,85,247,0.4);font-family:'Orbitron',sans-serif!important;
    text-transform:uppercase;letter-spacing:2px;font-weight:700;transition:all 0.3s ease;}
.stButton>button:hover,.stDownloadButton>button:hover{
    background:#a855f7!important;box-shadow:0 0 25px #a855f7,0 0 50px rgba(168,85,247,0.4)!important;
    transform:translateY(-2px);}
[data-testid="stFileUploader"]{border:2px dashed rgba(168,85,247,0.6)!important;
    border-radius:16px;background:rgba(168,85,247,0.04);
    box-shadow:0 0 20px rgba(168,85,247,0.15);transition:all 0.3s ease;}
hr{border-color:rgba(168,85,247,0.25)!important;}
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:1rem 0 1.5rem 0;'>
<h1>⬡ VECTRATRACK ML ⬡</h1>
<p style='color:#a78bfa;font-size:0.9rem;letter-spacing:3px;'>
BYTETRACK 2-STAGE · KALMAN FILTER · LINE CROSSING · HEATMAP
</p></div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ PARAMETERS")
    st.markdown("---")
    tracker_choice = st.selectbox("🧠 Algorithm",
        ["Custom SORT + ByteTrack 2-Stage", "ByteTrack (YOLOv8)", "BoT-SORT (YOLOv8)"])
    st.markdown("---")

    st.markdown("### 🎯 Detection")
    conf_high = st.slider("High-Conf Threshold", 0.2, 0.95, 0.50, 0.05,
                          help="Detections ≥ this → Stage-1 association")
    conf_low  = st.slider("Low-Conf Threshold",  0.05, 0.45, 0.10, 0.05,
                          help="Detections between low & high → Stage-2 (occlusion rescue)")
    iou_thr   = st.slider("IoU Threshold", 0.1, 0.9, 0.3, 0.05)
    st.markdown("---")

    st.markdown("### 🛤️ Trajectory")
    show_trail  = st.toggle("Past Trail", value=True)
    show_future = st.toggle("Future Prediction", value=True)
    trail_len   = st.slider("Trail Length (frames)", 5, 60, 30)
    show_occluded = st.toggle("Show Occluded Tracks", value=True,
                              help="Show tracks kept alive by low-conf detections (dashed box)")
    st.markdown("---")

    st.markdown("### 📏 Line Crossing Counter")
    enable_line = st.toggle("Enable Counting Line", value=False)
    if enable_line:
        st.caption("Set line endpoints as % of video width/height (0–100):")
        lx1 = st.slider("Line X1 %", 0, 100, 10)
        ly1 = st.slider("Line Y1 %", 0, 100, 50)
        lx2 = st.slider("Line X2 %", 0, 100, 90)
        ly2 = st.slider("Line Y2 %", 0, 100, 50)
    else:
        lx1, ly1, lx2, ly2 = 10, 50, 90, 50
    st.markdown("---")

    st.markdown("### 🌡️ Heatmap")
    show_heatmap   = st.toggle("Trajectory Heatmap Overlay", value=False)
    heatmap_alpha  = st.slider("Heatmap Opacity", 0.1, 0.9, 0.55, 0.05)
    st.markdown("---")

    st.markdown("### 🔬 Filters")
    class_filter = st.multiselect("Filter Classes (empty=ALL)", COCO_CLASSES, default=[])
    st.markdown("---")
    st.markdown("<div style='font-size:0.7rem;color:#6b21a8;text-align:center;line-height:1.8;'>VectraTrack ML v3.0<br>ByteTrack · Kalman · Heatmap<br>Line Crossing · CSV Export</div>",
                unsafe_allow_html=True)

# ── Model ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ── Overlay Drawing ───────────────────────────────────────────────────────────
def draw_cyber_overlay(frame, tracks, show_trail, show_future,
                       trail_len, show_occluded):
    h, w = frame.shape[:2]
    for trk in tracks:
        cls_id   = max(0, trk.get("cls", 0))
        colour   = get_class_colour(cls_id)
        occ      = trk.get("occluded", False)

        if occ and not show_occluded:
            continue

        x1,y1,x2,y2 = trk["bbox"]
        speed    = trk.get("speed", 0.0)
        cls_name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else "obj"

        # Past trail
        if show_trail and len(trk.get("history", [])) > 1:
            hist = trk["history"][-trail_len:]
            for i in range(1, len(hist)):
                a = i / len(hist)
                c = tuple(int(v * a) for v in colour)
                cv2.line(frame,
                         (int(hist[i-1][0]), int(hist[i-1][1])),
                         (int(hist[i][0]),   int(hist[i][1])),
                         c, max(1, int(a*3)), cv2.LINE_AA)

        # Future projection dashes
        if show_future and not occ and len(trk.get("future", [])) > 1:
            fut = trk["future"]
            for i in range(1, len(fut)):
                if i % 2 == 0: continue
                cv2.line(frame,
                         (int(fut[i-1][0]), int(fut[i-1][1])),
                         (int(fut[i][0]),   int(fut[i][1])),
                         (255, 255, 255), 1, cv2.LINE_AA)

        # Bounding box — dashed for occluded tracks
        box_col = (120, 120, 120) if occ else colour
        if occ:
            # Draw dashed rectangle for occluded
            for (ax,ay,bx,by) in [(x1,y1,x2,y1),(x2,y1,x2,y2),
                                   (x2,y2,x1,y2),(x1,y2,x1,y1)]:
                pts = np.linspace(0, 1, 10)
                for j in range(0, len(pts)-1, 2):
                    pa = (int(ax + (bx-ax)*pts[j]),   int(ay + (by-ay)*pts[j]))
                    pb = (int(ax + (bx-ax)*pts[j+1]), int(ay + (by-ay)*pts[j+1]))
                    cv2.line(frame, pa, pb, box_col, 1, cv2.LINE_AA)
        else:
            cv2.rectangle(frame, (x1-2,y1-2),(x2+2,y2+2),
                          tuple(c//2 for c in colour), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x1,y1),(x2,y2), colour, 2, cv2.LINE_AA)
            # Corner brackets
            bl = min(18, (x2-x1)//4, (y2-y1)//4)
            for px,py,dx,dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
                cv2.line(frame,(px,py),(px+dx*bl,py),colour,2,cv2.LINE_AA)
                cv2.line(frame,(px,py),(px,py+dy*bl),colour,2,cv2.LINE_AA)

        # Label
        occ_tag = " [OCC]" if occ else ""
        label = f"#{trk['id']} {cls_name} {speed:.1f}px/f{occ_tag}"
        (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        lx, ly = x1, max(y1-8, 12)
        cv2.rectangle(frame,(lx-2,ly-th-4),(lx+tw+4,ly+2),(10,5,25),cv2.FILLED)
        cv2.putText(frame, label,(lx,ly),cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, box_col, 1, cv2.LINE_AA)

    cv2.putText(frame, f"VECTRATRACK v3.0  |  {len(tracks)} ACTIVE",
                (10,22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (168,85,247), 1, cv2.LINE_AA)
    return frame

# ── Core Video Processor ──────────────────────────────────────────────────────
def process_video(video_path, tracker_name, conf_high, conf_low, iou_thr,
                  show_trail, show_future, trail_len, show_occluded,
                  enable_line, lx1, ly1, lx2, ly2,
                  show_heatmap, heatmap_alpha, class_filter):

    cap      = cv2.VideoCapture(video_path)
    fps_src  = cap.get(cv2.CAP_PROP_FPS) or 25
    width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tmp      = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out_path = tmp.name; tmp.close()
    writer   = cv2.VideoWriter(out_path,
                               cv2.VideoWriter_fourcc(*"mp4v"),
                               fps_src, (width, height))

    # Initialise tracker
    custom_tracker = None
    if "Custom SORT" in tracker_name:
        custom_tracker = CustomSortTracker(
            max_age=30, min_hits=3,
            iou_threshold=iou_thr,
            high_conf=conf_high, low_conf=conf_low,
        )

    # Initialise analytics
    line_counter = None
    if enable_line:
        pt1 = (int(lx1 / 100 * width),  int(ly1 / 100 * height))
        pt2 = (int(lx2 / 100 * width),  int(ly2 / 100 * height))
        line_counter = LineCrossingCounter(pt1, pt2, cooldown_frames=8)

    heatmap_acc = HeatmapAccumulator(width, height, blur_radius=51)
    telem       = TelemetryLogger(COCO_CLASSES)

    filter_ids = set()
    if class_filter:
        for n in class_filter:
            if n in COCO_CLASSES: filter_ids.add(COCO_CLASSES.index(n))

    telemetry  = []
    frame_idx  = 0
    total_time = 0.0
    progress   = st.progress(0, text="Initialising…")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        t0 = time.perf_counter()

        # ── Detection + Tracking ──────────────────────────────────────────
        if custom_tracker is not None:
            results = model.predict(frame, conf=conf_low, verbose=False)
            boxes   = results[0].boxes
            dets = []
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    if filter_ids and cls_id not in filter_ids: continue
                    x1,y1,x2,y2 = map(float, box.xyxy[0])
                    dets.append([x1,y1,x2,y2, float(box.conf[0]), cls_id])
            dets_np = np.array(dets) if dets else np.empty((0,6))
            tracks  = custom_tracker.update(dets_np)
        else:
            tyaml   = "bytetrack.yaml" if "ByteTrack" in tracker_name else "botsort.yaml"
            results = model.track(frame, persist=True, conf=conf_high,
                                  tracker=tyaml, verbose=False)
            boxes   = results[0].boxes
            tracks  = []
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    if box.id is None: continue
                    cls_id = int(box.cls[0])
                    if filter_ids and cls_id not in filter_ids: continue
                    x1,y1,x2,y2 = map(int, box.xyxy[0])
                    cx, cy = (x1+x2)/2, (y1+y2)/2
                    tracks.append({"id":int(box.id[0]),"bbox":[x1,y1,x2,y2],
                                   "cls":cls_id,"conf":round(float(box.conf[0]),2),
                                   "speed":0.0,"history":[],"future":[],"age":1,
                                   "occluded":False,"centre":(cx,cy)})

        elapsed    = time.perf_counter() - t0
        total_time += elapsed
        frame_fps  = 1.0 / elapsed if elapsed > 0 else 0

        # ── Analytics ────────────────────────────────────────────────────
        heatmap_acc.update(tracks)
        telem.log_frame(frame_idx, tracks, fps=frame_fps)

        cross_events = []
        if line_counter is not None:
            cross_events = line_counter.update(tracks, frame_idx, COCO_CLASSES)
            telem.log_crossings(cross_events)

        for trk in tracks:
            cls_name = COCO_CLASSES[trk["cls"]] if 0<=trk.get("cls",-1)<len(COCO_CLASSES) else "unknown"
            telemetry.append({"frame":frame_idx,"id":trk["id"],"class":cls_name,
                               "conf":trk.get("conf",0.0),"speed_px_per_frame":trk.get("speed",0.0),
                               "occluded":int(trk.get("occluded",False))})

        # ── Draw ─────────────────────────────────────────────────────────
        frame = draw_cyber_overlay(frame, tracks, show_trail, show_future,
                                   trail_len, show_occluded)

        if show_heatmap:
            frame = heatmap_acc.render_overlay(frame, alpha=heatmap_alpha)

        if line_counter is not None:
            frame = line_counter.draw(frame, cross_events)

        # FPS HUD
        cv2.putText(frame, f"FPS: {frame_fps:.1f}", (width-115,22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,170), 1, cv2.LINE_AA)

        writer.write(frame)
        frame_idx += 1
        progress.progress(min(frame_idx/max(n_frames,1), 1.0),
                          text=f"Frame {frame_idx}/{n_frames}  —  {frame_fps:.1f} FPS")

    cap.release(); writer.release(); progress.empty()

    timing = {
        "total_frames": frame_idx,
        "total_time_s": round(total_time, 2),
        "avg_fps":      round(frame_idx/total_time, 1) if total_time > 0 else 0,
        "unique_ids":   len(set(r["id"] for r in telemetry)),
        "total_in":     line_counter.total_in  if line_counter else 0,
        "total_out":    line_counter.total_out if line_counter else 0,
    }
    return out_path, telemetry, timing, telem, heatmap_acc, line_counter

# ── Analytics Display ─────────────────────────────────────────────────────────
def display_analytics(telemetry, timing, telem, heatmap_acc, line_counter):
    st.markdown("---")
    st.markdown("## 📊 TRACKING TELEMETRY")

    # ─ Summary metrics ─
    cols = st.columns(6)
    cols[0].metric("⏱ Avg FPS",       f"{timing['avg_fps']}")
    cols[1].metric("🎯 Unique IDs",    f"{timing['unique_ids']}")
    cols[2].metric("🎞 Frames",        f"{timing['total_frames']}")
    cols[3].metric("⚙️ Time",           f"{timing['total_time_s']}s")
    cols[4].metric("➡️ IN",             f"{timing['total_in']}")
    cols[5].metric("⬅️ OUT",            f"{timing['total_out']}")

    if not telemetry:
        st.warning("No objects tracked."); return

    df = pd.DataFrame(telemetry)

    # ─ Count over time ─
    st.markdown("### 📈 Object Count Over Time")
    cot = df.groupby("frame")["id"].count().reset_index()
    cot.columns = ["Frame","Count"]
    st.line_chart(cot.set_index("Frame"), color="#a855f7")

    ca, cb, cc = st.columns(3)
    with ca:
        st.markdown("### 🔮 Class Distribution")
        cc_df = df["class"].value_counts().reset_index()
        cc_df.columns = ["Class","Count"]
        st.bar_chart(cc_df.set_index("Class"))

    with cb:
        st.markdown("### ⚡ Avg Speed by Track")
        sp = df[["id","speed_px_per_frame"]].groupby("id").mean().reset_index()
        sp.columns = ["Track ID","Avg Speed"]
        st.bar_chart(sp.set_index("Track ID"), color="#00ffaa")

    with cc:
        if line_counter and line_counter.events:
            st.markdown("### 🚦 Crossing Events")
            ce_df = pd.DataFrame(line_counter.events)
            st.dataframe(ce_df, use_container_width=True)
        else:
            st.markdown("### 🚦 Crossing Events")
            st.info("Enable the counting line to see crossing data.")

    # ─ Heatmap section ─
    st.markdown("---")
    st.markdown("### 🌡️ Trajectory Heatmap")
    heat_png = heatmap_acc.get_png_bytes()
    if heatmap_acc.total_points > 0:
        st.image(heat_png, caption=f"Density heatmap — {heatmap_acc.total_points} positions recorded",
                 use_container_width=True)
        st.download_button("⬇️ Download Heatmap PNG", heat_png,
                           "vectratrack_heatmap.png", "image/png",
                           use_container_width=True)
    else:
        st.info("Heatmap will appear here after processing.")

    # ─ CSV Downloads ─
    st.markdown("---")
    st.markdown("### 🗂️ Export Data")
    dc1, dc2 = st.columns(2)
    with dc1:
        tracks_csv = telem.get_tracks_csv()
        st.download_button("⬇️ Tracking Telemetry CSV",
                           tracks_csv, "vectratrack_telemetry.csv", "text/csv",
                           use_container_width=True)
    with dc2:
        cross_csv = telem.get_crossings_csv()
        if cross_csv:
            st.download_button("⬇️ Line Crossing Events CSV",
                               cross_csv, "vectratrack_crossings.csv", "text/csv",
                               use_container_width=True)
        else:
            st.button("⬇️ Crossing Events CSV (no data)", disabled=True,
                      use_container_width=True)

# ── Main App ─────────────────────────────────────────────────────────────────
st.markdown("### 📂 Upload Video")
uploaded_file = st.file_uploader("Drop a video (MP4, AVI, MOV, MKV)…",
                                  type=["mp4","avi","mov","mkv"],
                                  label_visibility="collapsed")

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read()); tfile.flush()
    video_path = tfile.name

    cv_, ci_ = st.columns([2,1])
    with cv_: st.video(video_path)
    with ci_:
        st.markdown("### ⚡ Active Config")
        algo_short = "Custom SORT + ByteTrack" if "Custom" in tracker_choice else tracker_choice
        st.info(
            f"**Algorithm:** {algo_short}\n\n"
            f"**High-Conf:** {conf_high}  |  **Low-Conf:** {conf_low}\n\n"
            f"**IoU:** {iou_thr}\n\n"
            f"**Line:** {'ON' if enable_line else 'OFF'}  |  "
            f"**Heatmap:** {'ON' if show_heatmap else 'OFF'}\n\n"
            f"**Occluded display:** {'ON' if show_occluded else 'OFF'}\n\n"
            f"**Classes:** {', '.join(class_filter) if class_filter else 'ALL'}"
        )
        if "Custom SORT" in tracker_choice:
            st.success("✅ ByteTrack 2-Stage + Kalman Filter")
        else:
            st.info("ℹ️ YOLOv8 built-in tracker")

    st.markdown("---")

    if st.button(f"▶ LAUNCH {tracker_choice.upper()}", use_container_width=True):
        with st.spinner(f"Processing with {tracker_choice}…"):
            out_path, telemetry, timing, telem_obj, heatmap_acc, line_ctr = process_video(
                video_path, tracker_choice, conf_high, conf_low, iou_thr,
                show_trail, show_future, trail_len, show_occluded,
                enable_line, lx1, ly1, lx2, ly2,
                show_heatmap, heatmap_alpha, class_filter,
            )

        st.success("✅ Processing complete!")
        cd_, cv2_ = st.columns([1,2])
        with cd_:
            with open(out_path,"rb") as f:
                st.download_button("⬇️ Download Output Video", f.read(),
                                   "vectratrack_output.mp4","video/mp4",
                                   use_container_width=True)
        with cv2_:
            try: st.video(out_path)
            except Exception: st.warning("Preview unavailable — download above.")

        display_analytics(telemetry, timing, telem_obj, heatmap_acc, line_ctr)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#4c1d95;font-size:0.8rem;padding:1rem;letter-spacing:2px;'>
VECTRATRACK ML v3.0 &nbsp;·&nbsp; ByteTrack 2-Stage &nbsp;·&nbsp; Kalman Filter
&nbsp;·&nbsp; Line Crossing &nbsp;·&nbsp; Heatmap &nbsp;·&nbsp; Streamlit
</div>
""", unsafe_allow_html=True)
