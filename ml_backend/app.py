"""
VectraTrack ML — Streamlit Web Application (v2.0)
==================================================
Upgraded features:
  - Algorithm Selector: ByteTrack / BoT-SORT / Custom SORT (Kalman + Hungarian)
  - Trajectory Prediction: Past trails (neon glow) + Future projection (dashed)
  - Real-time Telemetry: FPS, object count, speed histogram, class breakdown
  - Multi-algorithm Benchmark: side-by-side timing comparison
  - Telemetry CSV Export: Frame-level log with ID, Class, Speed, BBox
  - Cyberpunk HUD overlay with per-class neon colours
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

st.set_page_config(page_title="VectraTrack ML", page_icon="🎯", layout="wide",
                   initial_sidebar_state="expanded")

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
    return palette[cls_id % len(palette)]

# ---- CSS -------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&display=swap');
header{visibility:hidden;} footer{visibility:hidden;} .stDeployButton{display:none;}
.stApp{background:radial-gradient(ellipse at 20% 20%,#0d0221 0%,#05010a 60%,#010010 100%);font-family:'Share Tech Mono',monospace;color:#d0c0f0;}
[data-testid="stSidebar"]{background:rgba(10,5,25,0.85)!important;border-right:1px solid rgba(168,85,247,0.3);backdrop-filter:blur(12px);}
[data-testid="stSidebar"]*{color:#c4b5fd!important;font-family:'Share Tech Mono',monospace!important;}
h1{font-family:'Orbitron',sans-serif!important;color:#fff!important;text-shadow:0 0 5px #fff,0 0 15px #a855f7,0 0 40px #a855f7,0 0 80px #7c3aed;letter-spacing:4px;text-transform:uppercase;text-align:center;}
h2,h3{font-family:'Orbitron',sans-serif!important;color:#c084fc!important;letter-spacing:2px;text-shadow:0 0 10px rgba(168,85,247,0.5);}
[data-testid="stMetric"]{background:rgba(168,85,247,0.08);border:1px solid rgba(168,85,247,0.25);border-radius:12px;padding:1rem;box-shadow:0 0 15px rgba(168,85,247,0.1);}
[data-testid="stMetricLabel"]{color:#a78bfa!important;font-size:0.75rem!important;}
[data-testid="stMetricValue"]{color:#fff!important;font-family:'Orbitron',sans-serif!important;}
.stButton>button,.stDownloadButton>button{background:linear-gradient(90deg,rgba(124,58,237,0.15) 0%,rgba(168,85,247,0.35) 100%)!important;color:#fff!important;border:1px solid #a855f7!important;border-radius:8px!important;box-shadow:0 0 12px rgba(168,85,247,0.4);font-family:'Orbitron',sans-serif!important;text-transform:uppercase;letter-spacing:2px;font-weight:700;transition:all 0.3s ease;}
.stButton>button:hover,.stDownloadButton>button:hover{background:#a855f7!important;box-shadow:0 0 25px #a855f7,0 0 50px rgba(168,85,247,0.4)!important;transform:translateY(-2px);}
[data-testid="stFileUploader"]{border:2px dashed rgba(168,85,247,0.6)!important;border-radius:16px;background:rgba(168,85,247,0.04);box-shadow:0 0 20px rgba(168,85,247,0.15);transition:all 0.3s ease;}
hr{border-color:rgba(168,85,247,0.25)!important;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center;padding:1rem 0 2rem 0;'>
<h1>⬡ VECTRATRACK ML ⬡</h1>
<p style='color:#a78bfa;font-size:1rem;letter-spacing:3px;'>
MULTI-OBJECT TRACKING · KALMAN FILTER · TRAJECTORY PREDICTION
</p></div>
""", unsafe_allow_html=True)

# ---- Sidebar ---------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ TRACKER PARAMETERS")
    st.markdown("---")
    tracker_choice = st.selectbox("🧠 Tracking Algorithm",
        ["Custom SORT (Kalman + Hungarian)", "ByteTrack", "BoT-SORT"])
    st.markdown("---")
    st.markdown("### 🎯 Detection")
    conf_threshold = st.slider("Confidence Threshold", 0.1, 0.95, 0.35, 0.05)
    iou_threshold  = st.slider("IoU Threshold (Custom SORT)", 0.1, 0.9, 0.3, 0.05)
    st.markdown("---")
    st.markdown("### 🛤️ Trajectory")
    show_trail   = st.toggle("Show Past Trail", value=True)
    show_future  = st.toggle("Show Future Prediction", value=True)
    trail_length = st.slider("Trail Length (frames)", 5, 60, 30)
    st.markdown("---")
    st.markdown("### 🔬 Filters")
    class_filter   = st.multiselect("Filter Classes (empty = ALL)", options=COCO_CLASSES, default=[])
    benchmark_mode = st.toggle("Benchmark Mode", value=False)
    st.markdown("---")
    st.markdown("<div style='font-size:0.75rem;color:#6b21a8;text-align:center;line-height:1.8;'>VectraTrack ML v2.0<br>Custom SORT · Kalman Filter<br>Hungarian Matcher · YOLOv8</div>",
                unsafe_allow_html=True)

# ---- Model -----------------------------------------------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ---- Drawing ---------------------------------------------------------------
def draw_cyber_overlay(frame, tracks, show_trail, show_future, trail_length):
    for trk in tracks:
        cls_id   = trk.get("cls", 0) if trk.get("cls", -1) >= 0 else 0
        colour   = get_class_colour(cls_id)
        trk_id   = trk["id"]
        x1,y1,x2,y2 = trk["bbox"]
        speed    = trk.get("speed", 0.0)
        cls_name = COCO_CLASSES[cls_id] if 0 <= cls_id < len(COCO_CLASSES) else "obj"

        if show_trail and len(trk.get("history", [])) > 1:
            hist = trk["history"][-trail_length:]
            for i in range(1, len(hist)):
                alpha = i / len(hist)
                c = tuple(int(v * alpha) for v in colour)
                cv2.line(frame, (int(hist[i-1][0]),int(hist[i-1][1])),
                         (int(hist[i][0]),int(hist[i][1])), c, max(1,int(alpha*3)), cv2.LINE_AA)

        if show_future and len(trk.get("future", [])) > 1:
            fut = trk["future"]
            for i in range(1, len(fut)):
                if i % 2 == 0: continue
                cv2.line(frame, (int(fut[i-1][0]),int(fut[i-1][1])),
                         (int(fut[i][0]),int(fut[i][1])), (255,255,255), 1, cv2.LINE_AA)

        cv2.rectangle(frame, (x1-2,y1-2),(x2+2,y2+2), tuple(c//2 for c in colour), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (x1,y1),(x2,y2), colour, 2, cv2.LINE_AA)

        bl = min(20,(x2-x1)//4,(y2-y1)//4)
        for px,py,dx,dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(frame,(px,py),(px+dx*bl,py),colour,2,cv2.LINE_AA)
            cv2.line(frame,(px,py),(px,py+dy*bl),colour,2,cv2.LINE_AA)

        label = f"#{trk_id} {cls_name} | {speed:.1f}px/f"
        (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        lx,ly = x1, max(y1-8,12)
        cv2.rectangle(frame,(lx-2,ly-th-4),(lx+tw+4,ly+2),(10,5,25),cv2.FILLED)
        cv2.putText(frame, label,(lx,ly),cv2.FONT_HERSHEY_SIMPLEX,0.45,colour,1,cv2.LINE_AA)

    cv2.putText(frame, f"VECTRATRACK ML  |  {len(tracks)} ACTIVE",
                (10,22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (168,85,247), 1, cv2.LINE_AA)
    return frame

# ---- Processing ------------------------------------------------------------
def process_video(video_path, tracker_name, conf, iou_thresh,
                  show_trail, show_future, trail_length, class_filter):
    cap = cv2.VideoCapture(video_path)
    fps      = cap.get(cv2.CAP_PROP_FPS) or 25
    width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out_path = temp_out.name; temp_out.close()
    writer   = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width,height))

    custom_tracker = None
    if tracker_name == "Custom SORT (Kalman + Hungarian)":
        custom_tracker = CustomSortTracker(max_age=30, min_hits=3, iou_threshold=iou_thresh)

    filter_ids = set()
    if class_filter:
        for n in class_filter:
            if n in COCO_CLASSES: filter_ids.add(COCO_CLASSES.index(n))

    telemetry, frame_idx, total_time = [], 0, 0.0
    progress = st.progress(0, text="Initialising tracker…")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        t_start = time.perf_counter()

        if custom_tracker is not None:
            results = model.predict(frame, conf=conf, verbose=False)
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
            tracker_yaml = "bytetrack.yaml" if "ByteTrack" in tracker_name else "botsort.yaml"
            results = model.track(frame, persist=True, conf=conf, tracker=tracker_yaml, verbose=False)
            boxes   = results[0].boxes
            tracks  = []
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    if box.id is None: continue
                    cls_id = int(box.cls[0])
                    if filter_ids and cls_id not in filter_ids: continue
                    x1,y1,x2,y2 = map(int, box.xyxy[0])
                    tracks.append({"id":int(box.id[0]),"bbox":[x1,y1,x2,y2],
                                   "cls":cls_id,"conf":round(float(box.conf[0]),2),
                                   "speed":0.0,"history":[],"future":[],"age":1})

        elapsed = time.perf_counter() - t_start
        total_time += elapsed
        frame_fps  = 1.0 / elapsed if elapsed > 0 else 0
        frame      = draw_cyber_overlay(frame, tracks, show_trail, show_future, trail_length)
        cv2.putText(frame, f"FPS: {frame_fps:.1f}", (width-120,22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,170), 1, cv2.LINE_AA)
        writer.write(frame)

        for trk in tracks:
            cls_name = COCO_CLASSES[trk["cls"]] if 0 <= trk.get("cls",-1) < len(COCO_CLASSES) else "unknown"
            telemetry.append({"frame":frame_idx,"id":trk["id"],"class":cls_name,
                               "conf":trk.get("conf",0.0),"x1":trk["bbox"][0],"y1":trk["bbox"][1],
                               "x2":trk["bbox"][2],"y2":trk["bbox"][3],
                               "speed_px_per_frame":trk.get("speed",0.0),"age_frames":trk.get("age",0)})
        frame_idx += 1
        progress.progress(min(frame_idx/max(n_frames,1),1.0),
                          text=f"Frame {frame_idx}/{n_frames}  —  {frame_fps:.1f} FPS")

    cap.release(); writer.release(); progress.empty()
    timing = {"total_frames":frame_idx,"total_time_s":round(total_time,2),
              "avg_fps":round(frame_idx/total_time,1) if total_time>0 else 0,
              "unique_ids":len(set(r["id"] for r in telemetry))}
    return out_path, telemetry, timing

# ---- Analytics -------------------------------------------------------------
def display_analytics(telemetry, timing, tracker_name):
    st.markdown("---")
    st.markdown("## 📊 TRACKING TELEMETRY")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("⏱ Avg FPS",         f"{timing['avg_fps']}")
    c2.metric("🎯 Unique IDs",      f"{timing['unique_ids']}")
    c3.metric("🎞 Total Frames",    f"{timing['total_frames']}")
    c4.metric("⚙️ Processing Time",  f"{timing['total_time_s']}s")

    if not telemetry:
        st.warning("No objects tracked in this video."); return

    df = pd.DataFrame(telemetry)
    st.markdown("### 📈 Object Count Over Time")
    cot = df.groupby("frame")["id"].count().reset_index()
    cot.columns = ["Frame","Object Count"]
    st.line_chart(cot.set_index("Frame"), color="#a855f7")

    ca,cb = st.columns(2)
    with ca:
        st.markdown("### 🔮 Class Distribution")
        cc = df["class"].value_counts().reset_index()
        cc.columns = ["Class","Count"]
        st.bar_chart(cc.set_index("Class"))
    with cb:
        st.markdown("### ⚡ Speed Distribution (px/frame)")
        sp = df[["id","speed_px_per_frame"]].groupby("id").mean().reset_index()
        sp.columns = ["Track ID","Avg Speed"]
        st.bar_chart(sp.set_index("Track ID"), color="#00ffaa")

    st.markdown("### 🗂️ Raw Telemetry Log")
    st.dataframe(df.head(500), use_container_width=True)
    st.download_button("⬇️ Download Telemetry CSV", df.to_csv(index=False).encode("utf-8"),
                       "vectratrack_telemetry.csv", "text/csv", use_container_width=True)

# ---- Benchmark -------------------------------------------------------------
def run_benchmark(video_path, conf, iou_thresh, class_filter):
    algorithms = ["Custom SORT (Kalman + Hungarian)","ByteTrack","BoT-SORT"]
    results = {}
    bp = st.progress(0)
    for i, algo in enumerate(algorithms):
        bp.progress(i/len(algorithms), text=f"Running {algo}…")
        _, _, timing = process_video(video_path, algo, conf, iou_thresh, False, False, 30, class_filter)
        results[algo] = timing
    bp.empty()
    return results

# ---- Main ------------------------------------------------------------------
st.markdown("### 📂 Upload Video for Tracking")
uploaded_file = st.file_uploader("Drop a video file here (MP4, AVI, MOV)…",
                                  type=["mp4","avi","mov","mkv"],
                                  label_visibility="collapsed")

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read()); tfile.flush()
    video_path = tfile.name

    cv, ci = st.columns([2,1])
    with cv: st.video(video_path)
    with ci:
        st.markdown("### ⚡ Quick Config")
        st.info(f"**Algorithm:** {tracker_choice}\n\n"
                f"**Confidence:** {conf_threshold}\n\n"
                f"**IoU:** {iou_threshold}\n\n"
                f"**Trail:** {'ON' if show_trail else 'OFF'}  |  "
                f"**Predict:** {'ON' if show_future else 'OFF'}\n\n"
                f"**Classes:** {', '.join(class_filter) if class_filter else 'ALL'}")
        if tracker_choice == "Custom SORT (Kalman + Hungarian)":
            st.success("✅ Custom Kalman Filter + Hungarian Matcher")

    st.markdown("---")

    if benchmark_mode:
        st.markdown("## 🏁 BENCHMARK MODE")
        st.warning("Runs all 3 trackers on the same video for fair FPS comparison.")
        if st.button("▶ RUN BENCHMARK", use_container_width=True):
            br = run_benchmark(video_path, conf_threshold, iou_threshold, class_filter)
            rows = [{"Algorithm":k,"Avg FPS":v["avg_fps"],"Total Time (s)":v["total_time_s"],
                     "Unique IDs":v["unique_ids"],"Frames":v["total_frames"]} for k,v in br.items()]
            bdf = pd.DataFrame(rows).set_index("Algorithm")
            st.dataframe(bdf, use_container_width=True)
            st.bar_chart(bdf["Avg FPS"])
    else:
        if st.button(f"▶ LAUNCH {tracker_choice.upper()}", use_container_width=True):
            with st.spinner(f"Running {tracker_choice}…"):
                out_path, telemetry, timing = process_video(
                    video_path, tracker_choice, conf_threshold, iou_threshold,
                    show_trail, show_future, trail_length, class_filter)

            st.success("✅ Tracking complete!")
            cd, cv2_ = st.columns([1,2])
            with cd:
                with open(out_path,"rb") as f:
                    st.download_button("⬇️ Download Output Video", f.read(),
                                       "vectratrack_output.mp4", "video/mp4", use_container_width=True)
            with cv2_:
                try: st.video(out_path)
                except Exception: st.warning("Preview unavailable — download above.")

            display_analytics(telemetry, timing, tracker_choice)

st.markdown("---")
st.markdown("<div style='text-align:center;color:#4c1d95;font-size:0.8rem;padding:1rem;letter-spacing:2px;'>VECTRATRACK ML v2.0 &nbsp;·&nbsp; YOLOv8 + Custom SORT (Kalman + Hungarian) &nbsp;·&nbsp; Trajectory Prediction &nbsp;·&nbsp; Streamlit</div>",
            unsafe_allow_html=True)
