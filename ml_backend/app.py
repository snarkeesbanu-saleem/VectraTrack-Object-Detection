"""
VectraTrack ML — Unified App (v4.0)
=====================================
Single Streamlit app combining:
  Tab 1 — 🎥 Live Camera   : Real-time webcam detection + tracking (streamlit-webrtc + YOLOv8)
  Tab 2 — 📹 Video Analysis: Upload video → Custom SORT/ByteTrack → annotated output
  Tab 3 — 📊 Analytics     : Heatmap, Crossing counter, CSV export

Replaces the separate React/Vercel frontend + Streamlit backend with ONE unified URL.
"""

import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import time
import pandas as pd
import threading
from custom_tracker import CustomSortTracker
from analytics import LineCrossingCounter, HeatmapAccumulator, TelemetryLogger

# streamlit-webrtc for real-time webcam
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
    import av
    WEBRTC_OK = True
except ImportError:
    WEBRTC_OK = False

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VectraTrack ML",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── COCO Classes ──────────────────────────────────────────────────────────────
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

def get_colour(cls_id):
    palette = [(0,255,170),(255,0,200),(0,220,255),(255,180,0),
               (180,0,255),(0,255,80),(255,80,0),(0,100,255)]
    return palette[int(cls_id) % len(palette)]

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&display=swap');
header{visibility:hidden;} footer{visibility:hidden;} .stDeployButton{display:none;}
.stApp{
    background:radial-gradient(ellipse at 20% 20%,#0d0221 0%,#05010a 60%,#010010 100%);
    font-family:'Share Tech Mono',monospace; color:#d0c0f0;
}
[data-testid="stSidebar"]{
    background:rgba(10,5,25,0.92)!important;
    border-right:1px solid rgba(168,85,247,0.3);
    backdrop-filter:blur(14px);
}
[data-testid="stSidebar"]*{color:#c4b5fd!important;font-family:'Share Tech Mono',monospace!important;}
h1{
    font-family:'Orbitron',sans-serif!important; color:#fff!important;
    text-shadow:0 0 5px #fff,0 0 20px #a855f7,0 0 50px #a855f7,0 0 90px #7c3aed;
    letter-spacing:4px; text-transform:uppercase; text-align:center;
}
h2,h3{font-family:'Orbitron',sans-serif!important; color:#c084fc!important;
      letter-spacing:2px; text-shadow:0 0 10px rgba(168,85,247,0.5);}
/* Tabs */
[data-testid="stTabs"] button{
    font-family:'Orbitron',sans-serif!important;
    color:#a78bfa!important; font-size:0.8rem!important;
    letter-spacing:1px; text-transform:uppercase;
    border-radius:8px 8px 0 0!important;
}
[data-testid="stTabs"] button[aria-selected="true"]{
    color:#fff!important;
    background:rgba(168,85,247,0.15)!important;
    border-bottom:2px solid #a855f7!important;
    box-shadow:0 0 12px rgba(168,85,247,0.3);
}
[data-testid="stMetric"]{
    background:rgba(168,85,247,0.08); border:1px solid rgba(168,85,247,0.25);
    border-radius:12px; padding:1rem; box-shadow:0 0 15px rgba(168,85,247,0.1);
}
[data-testid="stMetricLabel"]{color:#a78bfa!important;font-size:0.75rem!important;}
[data-testid="stMetricValue"]{color:#fff!important;font-family:'Orbitron',sans-serif!important;}
.stButton>button,.stDownloadButton>button{
    background:linear-gradient(90deg,rgba(124,58,237,0.15) 0%,rgba(168,85,247,0.35) 100%)!important;
    color:#fff!important; border:1px solid #a855f7!important; border-radius:8px!important;
    box-shadow:0 0 12px rgba(168,85,247,0.4); font-family:'Orbitron',sans-serif!important;
    text-transform:uppercase; letter-spacing:2px; font-weight:700; transition:all 0.3s ease;
}
.stButton>button:hover,.stDownloadButton>button:hover{
    background:#a855f7!important;
    box-shadow:0 0 25px #a855f7,0 0 50px rgba(168,85,247,0.4)!important;
    transform:translateY(-2px);
}
[data-testid="stFileUploader"]{
    border:2px dashed rgba(168,85,247,0.6)!important; border-radius:16px;
    background:rgba(168,85,247,0.04); box-shadow:0 0 20px rgba(168,85,247,0.15);
}
hr{border-color:rgba(168,85,247,0.25)!important;}
/* WebRTC video container */
.stVideo video, video{border-radius:12px; border:1px solid rgba(168,85,247,0.3);}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding:0.8rem 0 1.2rem;'>
<h1>⬡ VECTRATRACK ML ⬡</h1>
<p style='color:#a78bfa; font-size:0.85rem; letter-spacing:3px;'>
LIVE CAMERA · VIDEO ANALYSIS · BYTETRACK 2-STAGE · HEATMAP · LINE CROSSING
</p>
</div>
""", unsafe_allow_html=True)

# ── Model loading (cached) ────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ GLOBAL SETTINGS")
    st.markdown("---")
    conf_high     = st.slider("High-Conf Threshold", 0.2, 0.95, 0.50, 0.05)
    conf_low      = st.slider("Low-Conf Threshold",  0.05, 0.45, 0.10, 0.05)
    iou_thr       = st.slider("IoU Threshold",       0.1,  0.9,  0.30, 0.05)
    st.markdown("---")
    show_trail    = st.toggle("Past Trail",           value=True)
    show_future   = st.toggle("Future Prediction",    value=True)
    trail_len     = st.slider("Trail Length",         5, 60, 30)
    show_occluded = st.toggle("Show Occluded Tracks", value=True)
    st.markdown("---")
    class_filter  = st.multiselect("Filter Classes (empty=ALL)", COCO_CLASSES, [])
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.7rem;color:#6b21a8;text-align:center;line-height:1.9;'>
    VectraTrack ML v4.0<br>
    🎥 Live Camera + 📹 Video<br>
    ByteTrack · Kalman · Heatmap<br>
    Line Counter · CSV Export
    </div>
    """, unsafe_allow_html=True)

# ── Overlay drawing helper ─────────────────────────────────────────────────────
def draw_overlay(frame, tracks, show_trail, show_future, trail_len, show_occluded):
    h, w = frame.shape[:2]
    for trk in tracks:
        cls_id = max(0, trk.get("cls", 0))
        colour = get_colour(cls_id)
        occ    = trk.get("occluded", False)
        if occ and not show_occluded:
            continue
        x1,y1,x2,y2 = trk["bbox"]
        speed    = trk.get("speed", 0.0)
        cls_name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else "obj"
        box_col  = (120,120,120) if occ else colour

        # Trail
        if show_trail and len(trk.get("history",[])) > 1:
            hist = trk["history"][-trail_len:]
            for i in range(1, len(hist)):
                a = i / len(hist)
                c = tuple(int(v*a) for v in colour)
                cv2.line(frame,(int(hist[i-1][0]),int(hist[i-1][1])),
                         (int(hist[i][0]),int(hist[i][1])),c,max(1,int(a*3)),cv2.LINE_AA)

        # Future projection
        if show_future and not occ and len(trk.get("future",[])) > 1:
            fut = trk["future"]
            for i in range(1,len(fut)):
                if i%2==0: continue
                cv2.line(frame,(int(fut[i-1][0]),int(fut[i-1][1])),
                         (int(fut[i][0]),int(fut[i][1])),(255,255,255),1,cv2.LINE_AA)

        # Box
        if occ:
            cv2.rectangle(frame,(x1,y1),(x2,y2),box_col,1,cv2.LINE_AA)
        else:
            cv2.rectangle(frame,(x1-2,y1-2),(x2+2,y2+2),
                          tuple(c//2 for c in colour),1,cv2.LINE_AA)
            cv2.rectangle(frame,(x1,y1),(x2,y2),colour,2,cv2.LINE_AA)
            bl = min(16,(x2-x1)//4,(y2-y1)//4)
            for px,py,dx,dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
                cv2.line(frame,(px,py),(px+dx*bl,py),colour,2,cv2.LINE_AA)
                cv2.line(frame,(px,py),(px,py+dy*bl),colour,2,cv2.LINE_AA)

        # Label
        occ_tag = " [OCC]" if occ else ""
        label = f"#{trk['id']} {cls_name} {speed:.1f}px/f{occ_tag}"
        (tw,th),_ = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.42,1)
        lx,ly = x1, max(y1-6,12)
        cv2.rectangle(frame,(lx-2,ly-th-4),(lx+tw+4,ly+2),(10,5,25),cv2.FILLED)
        cv2.putText(frame,label,(lx,ly),cv2.FONT_HERSHEY_SIMPLEX,0.42,box_col,1,cv2.LINE_AA)

    cv2.putText(frame,f"VECTRATRACK  |  {len(tracks)} ACTIVE",
                (8,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(168,85,247),1,cv2.LINE_AA)
    return frame

# ── Filter class IDs ──────────────────────────────────────────────────────────
def get_filter_ids(class_filter):
    ids = set()
    for n in class_filter:
        if n in COCO_CLASSES: ids.add(COCO_CLASSES.index(n))
    return ids

# =============================================================================
# TAB 1 — LIVE CAMERA
# =============================================================================

class LiveTracker:
    """Thread-safe stateful tracker for WebRTC stream."""
    def __init__(self):
        self.tracker  = CustomSortTracker(max_age=30, min_hits=2,
                                          iou_threshold=0.3,
                                          high_conf=0.5, low_conf=0.1)
        self.lock     = threading.Lock()
        self.stats    = {"fps": 0.0, "count": 0, "total": 0}
        self._last_t  = time.perf_counter()

    def process(self, frame_bgr, conf_high, conf_low, iou,
                show_trail, show_future, trail_len, show_occluded, filter_ids):
        t0      = time.perf_counter()
        results = model.predict(frame_bgr, conf=conf_low, verbose=False)
        boxes   = results[0].boxes
        dets    = []
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cid = int(box.cls[0])
                if filter_ids and cid not in filter_ids: continue
                x1,y1,x2,y2 = map(float, box.xyxy[0])
                dets.append([x1,y1,x2,y2,float(box.conf[0]),cid])

        dets_np = np.array(dets) if dets else np.empty((0,6))
        with self.lock:
            self.tracker.high_conf = conf_high
            self.tracker.low_conf  = conf_low
            self.tracker.iou_threshold = iou
            tracks = self.tracker.update(dets_np)

        frame_out = draw_overlay(frame_bgr.copy(), tracks,
                                 show_trail, show_future, trail_len, show_occluded)
        elapsed   = time.perf_counter() - t0
        fps       = 1.0/elapsed if elapsed > 0 else 0
        cv2.putText(frame_out, f"FPS: {fps:.1f}",
                    (frame_out.shape[1]-100, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,170), 1, cv2.LINE_AA)
        with self.lock:
            self.stats = {"fps": round(fps,1), "count": len(tracks),
                          "total": self.tracker.frame_count}
        return frame_out

def render_live_tab(conf_high, conf_low, iou_thr,
                    show_trail, show_future, trail_len, show_occluded, class_filter):
    st.markdown("## 🎥 LIVE CAMERA TRACKER")
    st.markdown("""
    <p style='color:#a78bfa; font-size:0.85rem;'>
    Real-time object detection & multi-object tracking directly from your webcam —
    same as the Vercel browser app, now powered by <b>YOLOv8 + ByteTrack 2-Stage</b>.
    </p>
    """, unsafe_allow_html=True)

    if not WEBRTC_OK:
        st.error("⚠️ `streamlit-webrtc` not installed. Run: `pip install streamlit-webrtc av`")
        return

    filter_ids = get_filter_ids(class_filter)

    # Initialise stateful tracker in session
    if "live_tracker" not in st.session_state:
        st.session_state.live_tracker = LiveTracker()
    lt = st.session_state.live_tracker

    c_stream, c_stats = st.columns([3, 1])

    with c_stream:
        RTC_CONFIG = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )

        def video_frame_callback(frame):
            img = frame.to_ndarray(format="bgr24")
            out = lt.process(img, conf_high, conf_low, iou_thr,
                             show_trail, show_future, trail_len, show_occluded, filter_ids)
            return av.VideoFrame.from_ndarray(out, format="bgr24")

        webrtc_streamer(
            key="live-tracker",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIG,
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
            async_processing=True,
        )

    with c_stats:
        st.markdown("### 📡 LIVE STATS")
        stats = lt.stats
        st.metric("⚡ FPS",        stats.get("fps", 0))
        st.metric("🎯 Active",     stats.get("count", 0))
        st.metric("🎞 Frames",     stats.get("total", 0))
        st.markdown("---")
        st.markdown("""
        **Tracker:** Custom SORT  
        **Detection:** YOLOv8 nano  
        **Algorithm:**  
        • Kalman Filter (7D state)  
        • Hungarian Matching  
        • ByteTrack 2-Stage  
        """)
        st.info("Allow camera access when prompted by the browser.")

# =============================================================================
# TAB 2 — VIDEO ANALYSIS
# =============================================================================

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
                               cv2.VideoWriter_fourcc(*"mp4v"), fps_src, (width,height))

    custom_tracker = None
    if "Custom" in tracker_name:
        custom_tracker = CustomSortTracker(max_age=30, min_hits=3,
                                           iou_threshold=iou_thr,
                                           high_conf=conf_high, low_conf=conf_low)

    line_counter = None
    if enable_line:
        pt1 = (int(lx1/100*width), int(ly1/100*height))
        pt2 = (int(lx2/100*width), int(ly2/100*height))
        line_counter = LineCrossingCounter(pt1, pt2, cooldown_frames=8)

    heatmap_acc = HeatmapAccumulator(width, height, blur_radius=51)
    telem       = TelemetryLogger(COCO_CLASSES)
    filter_ids  = get_filter_ids(class_filter)
    telemetry   = []
    frame_idx   = 0
    total_time  = 0.0
    progress    = st.progress(0, text="Initialising tracker…")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        t0 = time.perf_counter()

        if custom_tracker is not None:
            results = model.predict(frame, conf=conf_low, verbose=False)
            boxes   = results[0].boxes
            dets = []
            if boxes is not None and len(boxes)>0:
                for box in boxes:
                    cid = int(box.cls[0])
                    if filter_ids and cid not in filter_ids: continue
                    x1,y1,x2,y2 = map(float, box.xyxy[0])
                    dets.append([x1,y1,x2,y2,float(box.conf[0]),cid])
            dets_np = np.array(dets) if dets else np.empty((0,6))
            tracks  = custom_tracker.update(dets_np)
        else:
            tyaml   = "bytetrack.yaml" if "ByteTrack" in tracker_name else "botsort.yaml"
            results = model.track(frame, persist=True, conf=conf_high,
                                  tracker=tyaml, verbose=False)
            boxes   = results[0].boxes; tracks = []
            if boxes is not None and len(boxes)>0:
                for box in boxes:
                    if box.id is None: continue
                    cid = int(box.cls[0])
                    if filter_ids and cid not in filter_ids: continue
                    x1,y1,x2,y2 = map(int, box.xyxy[0])
                    tracks.append({"id":int(box.id[0]),"bbox":[x1,y1,x2,y2],
                                   "cls":cid,"conf":round(float(box.conf[0]),2),
                                   "speed":0.0,"history":[],"future":[],"age":1,
                                   "occluded":False,"centre":((x1+x2)/2,(y1+y2)/2)})

        elapsed    = time.perf_counter()-t0
        total_time += elapsed
        frame_fps  = 1.0/elapsed if elapsed>0 else 0

        heatmap_acc.update(tracks)
        telem.log_frame(frame_idx, tracks, fps=frame_fps)
        cross_ev = []
        if line_counter:
            cross_ev = line_counter.update(tracks, frame_idx, COCO_CLASSES)
            telem.log_crossings(cross_ev)

        for trk in tracks:
            cn = COCO_CLASSES[trk["cls"]] if 0<=trk.get("cls",-1)<len(COCO_CLASSES) else "unknown"
            telemetry.append({"frame":frame_idx,"id":trk["id"],"class":cn,
                               "conf":trk.get("conf",0.0),"speed":trk.get("speed",0.0),
                               "occluded":int(trk.get("occluded",False))})

        frame = draw_overlay(frame, tracks, show_trail, show_future,
                             trail_len, show_occluded)
        if show_heatmap:
            frame = heatmap_acc.render_overlay(frame, alpha=heatmap_alpha)
        if line_counter:
            frame = line_counter.draw(frame, cross_ev)
        cv2.putText(frame, f"FPS:{frame_fps:.1f}",
                    (width-100,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,170),1,cv2.LINE_AA)

        writer.write(frame)
        frame_idx += 1
        progress.progress(min(frame_idx/max(n_frames,1),1.0),
                          text=f"Frame {frame_idx}/{n_frames}  —  {frame_fps:.1f} FPS")

    cap.release(); writer.release(); progress.empty()
    timing = {"total_frames":frame_idx,"total_time_s":round(total_time,2),
              "avg_fps":round(frame_idx/total_time,1) if total_time>0 else 0,
              "unique_ids":len(set(r["id"] for r in telemetry)),
              "total_in":line_counter.total_in if line_counter else 0,
              "total_out":line_counter.total_out if line_counter else 0}
    return out_path, telemetry, timing, telem, heatmap_acc, line_counter


def render_video_tab(conf_high, conf_low, iou_thr,
                     show_trail, show_future, trail_len, show_occluded, class_filter):
    st.markdown("## 📹 VIDEO ANALYSIS")

    # Sub-settings specific to this tab
    c1, c2 = st.columns(2)
    with c1:
        tracker_choice = st.selectbox("🧠 Algorithm",
            ["Custom SORT + ByteTrack 2-Stage", "ByteTrack (YOLOv8)", "BoT-SORT (YOLOv8)"])
    with c2:
        show_heatmap  = st.toggle("🌡️ Heatmap Overlay", value=False)
        heatmap_alpha = st.slider("Heatmap Opacity", 0.1, 0.9, 0.55, 0.05)

    # Line crossing
    with st.expander("📏 Line Crossing Counter", expanded=False):
        enable_line = st.toggle("Enable Counting Line", value=False)
        if enable_line:
            lc1, lc2 = st.columns(2)
            with lc1:
                lx1 = st.slider("X1 %", 0, 100, 10)
                ly1 = st.slider("Y1 %", 0, 100, 50)
            with lc2:
                lx2 = st.slider("X2 %", 0, 100, 90)
                ly2 = st.slider("Y2 %", 0, 100, 50)
        else:
            lx1,ly1,lx2,ly2 = 10,50,90,50

    uploaded = st.file_uploader("Drop a video (MP4, AVI, MOV, MKV)…",
                                type=["mp4","avi","mov","mkv"],
                                label_visibility="collapsed")
    if uploaded is None:
        st.info("👆 Upload a video above to begin analysis.")
        return

    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded.read()); tfile.flush()
    video_path = tfile.name

    cv_, ci_ = st.columns([2,1])
    with cv_: st.video(video_path)
    with ci_:
        st.markdown("### ⚡ Config")
        st.info(
            f"**Algorithm:** {tracker_choice}\n\n"
            f"**High-Conf:** {conf_high}  |  **Low-Conf:** {conf_low}\n\n"
            f"**Line:** {'ON' if enable_line else 'OFF'}  |  "
            f"**Heatmap:** {'ON' if show_heatmap else 'OFF'}"
        )

    st.markdown("---")
    if st.button(f"▶ LAUNCH {tracker_choice.upper()}", use_container_width=True):
        with st.spinner("Processing…"):
            out_path, telemetry, timing, telem_obj, hmap, line_ctr = process_video(
                video_path, tracker_choice, conf_high, conf_low, iou_thr,
                show_trail, show_future, trail_len, show_occluded,
                enable_line, lx1, ly1, lx2, ly2,
                show_heatmap, heatmap_alpha, class_filter,
            )

        st.success("✅ Done!")
        cd_, cv2_ = st.columns([1,2])
        with cd_:
            with open(out_path,"rb") as f:
                st.download_button("⬇️ Output Video", f.read(),
                                   "vectratrack_output.mp4","video/mp4",
                                   use_container_width=True)
        with cv2_:
            try: st.video(out_path)
            except Exception: st.warning("Preview unavailable — download above.")

        # ── Analytics summary ──
        st.markdown("---")
        st.markdown("### 📊 RESULTS")
        cols = st.columns(6)
        cols[0].metric("⏱ Avg FPS",    timing["avg_fps"])
        cols[1].metric("🎯 Unique IDs", timing["unique_ids"])
        cols[2].metric("🎞 Frames",     timing["total_frames"])
        cols[3].metric("⚙️ Time",        f"{timing['total_time_s']}s")
        cols[4].metric("➡️ IN",          timing["total_in"])
        cols[5].metric("⬅️ OUT",         timing["total_out"])

        if telemetry:
            df = pd.DataFrame(telemetry)
            ca,cb = st.columns(2)
            with ca:
                st.markdown("#### 📈 Objects Over Time")
                cot = df.groupby("frame")["id"].count().reset_index()
                cot.columns=["Frame","Count"]
                st.line_chart(cot.set_index("Frame"), color="#a855f7")
            with cb:
                st.markdown("#### 🔮 Class Distribution")
                cc = df["class"].value_counts().reset_index()
                cc.columns=["Class","Count"]
                st.bar_chart(cc.set_index("Class"))

        # Heatmap download
        if hmap and hmap.total_points > 0:
            st.markdown("#### 🌡️ Trajectory Heatmap")
            heat_png = hmap.get_png_bytes()
            st.image(heat_png, caption=f"{hmap.total_points} positions", use_container_width=True)
            st.download_button("⬇️ Download Heatmap PNG", heat_png,
                               "vectratrack_heatmap.png","image/png", use_container_width=True)

        # Crossing events
        if line_ctr and line_ctr.events:
            st.markdown("#### 🚦 Crossing Events")
            st.dataframe(pd.DataFrame(line_ctr.events), use_container_width=True)

        # CSV exports
        st.markdown("#### 🗂️ Export")
        d1,d2 = st.columns(2)
        with d1:
            st.download_button("⬇️ Telemetry CSV", telem_obj.get_tracks_csv(),
                               "telemetry.csv","text/csv", use_container_width=True)
        with d2:
            cross_csv = telem_obj.get_crossings_csv()
            if cross_csv:
                st.download_button("⬇️ Crossings CSV", cross_csv,
                                   "crossings.csv","text/csv", use_container_width=True)


# =============================================================================
# TAB 3 — ABOUT / FEATURE GUIDE
# =============================================================================

def render_about_tab():
    st.markdown("## 📖 FEATURES & ALGORITHMS")

    st.markdown("""
    <div style='background:rgba(168,85,247,0.06);border:1px solid rgba(168,85,247,0.2);
    border-radius:16px;padding:1.5rem;'>
    """, unsafe_allow_html=True)

    cols = st.columns(3)
    with cols[0]:
        st.markdown("""
        ### 🎥 Live Camera
        - Real-time webcam tracking
        - YOLOv8 nano (≤30 FPS)
        - Multi-object tracking w/ IDs
        - Motion trails + future prediction
        - Live FPS / count stats
        """)
    with cols[1]:
        st.markdown("""
        ### 📹 Video Analysis
        - Upload MP4/AVI/MOV/MKV
        - 3 algorithm choices
        - Heatmap overlay (TURBO colormap)
        - Virtual counting line (IN/OUT)
        - Download annotated video
        """)
    with cols[2]:
        st.markdown("""
        ### 📊 Export & Analytics
        - Object count over time chart
        - Class distribution bar chart
        - Telemetry CSV (per-frame)
        - Crossing events CSV
        - Heatmap PNG download
        """)

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 🧠 Algorithm Details")

    tab_a, tab_b, tab_c = st.tabs(["Kalman Filter", "ByteTrack 2-Stage", "Line Crossing"])
    with tab_a:
        st.markdown(r"""
**State vector (7D):** $x = [c_x,\ c_y,\ s,\ r,\ \dot{c}_x,\ \dot{c}_y,\ \dot{s}]^T$

**Predict step:**
$$x_{k|k-1} = F\,x_{k-1}$$
$$P_{k|k-1} = F\,P_{k-1}\,F^T + Q$$

**Update step:**
$$K = P_{k|k-1}\,H^T\,(H\,P_{k|k-1}\,H^T + R)^{-1}$$
$$x_k = x_{k|k-1} + K(z - H\,x_{k|k-1})$$
        """)
    with tab_b:
        st.markdown("""
**Two-stage detection split:**

| Stage | Detections | Matched to | IoU Gate |
|---|---|---|---|
| 1 | conf ≥ high_thresh | All active tracks | Strict |
| 2 | low_thresh ≤ conf < high_thresh | **Unmatched** tracks only | Relaxed |
| ✗ | Orphan low-conf | Not matched → **discarded** | — |

Stage 2 rescues tracks through occlusion without spawning ghost tracks.
        """)
    with tab_c:
        st.markdown("""
**Cross-product side detection:**

For a line $P_1 → P_2$ and track centre $C$:

$$\\text{side} = (P_2 - P_1) \\times (C - P_1)$$

- $> 0$ → side A ("IN")  
- $< 0$ → side B ("OUT")

A crossing is detected when the sign changes between consecutive frames.
Cooldown prevents double-counting the same object.
        """)

    st.markdown("---")
    st.markdown("""
    <div style='text-align:center; color:#6b21a8; font-size:0.8rem; letter-spacing:2px;'>
    VECTRATRACK ML v4.0 &nbsp;·&nbsp;
    <a href='https://github.com/snarkeesbanu-saleem/CodeAlpha_VectraTrack-Object-Detection'
       style='color:#a855f7;'>GitHub</a> &nbsp;·&nbsp;
    Built with YOLOv8 · Streamlit · OpenCV · SciPy
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# MAIN — TABS
# =============================================================================

tab1, tab2, tab3 = st.tabs([
    "🎥  Live Camera",
    "📹  Video Analysis",
    "📖  Features & Algorithms",
])

with tab1:
    render_live_tab(conf_high, conf_low, iou_thr,
                    show_trail, show_future, trail_len, show_occluded, class_filter)

with tab2:
    render_video_tab(conf_high, conf_low, iou_thr,
                     show_trail, show_future, trail_len, show_occluded, class_filter)

with tab3:
    render_about_tab()
