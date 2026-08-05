"""
VectraTrack CLI (v2.0) — Multi-Object Tracking from the command line.
Supports ByteTrack, BoT-SORT, and Custom SORT (Kalman + Hungarian).
"""

import cv2, argparse, time, numpy as np
from ultralytics import YOLO
from custom_tracker import CustomSortTracker

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
NEON = [(0,255,170),(255,0,200),(0,220,255),(255,180,0),(180,0,255),(0,255,80),(255,80,0),(0,100,255)]

def col(cls_id): return NEON[cls_id % len(NEON)]

def draw_track(frame, trk, show_trail, show_future, trail_len=30):
    cls_id = trk.get("cls",0) if trk.get("cls",-1) >= 0 else 0
    colour = col(cls_id)
    x1,y1,x2,y2 = trk["bbox"]
    speed    = trk.get("speed",0.0)
    cls_name = COCO_CLASSES[cls_id] if 0 <= cls_id < len(COCO_CLASSES) else "obj"

    if show_trail and len(trk.get("history",[])) > 1:
        hist = trk["history"][-trail_len:]
        for i in range(1,len(hist)):
            a = i/len(hist)
            c = tuple(int(v*a) for v in colour)
            cv2.line(frame,(int(hist[i-1][0]),int(hist[i-1][1])),(int(hist[i][0]),int(hist[i][1])),c,max(1,int(a*3)),cv2.LINE_AA)

    if show_future and len(trk.get("future",[])) > 1:
        fut = trk["future"]
        for i in range(1,len(fut)):
            if i%2==0: continue
            cv2.line(frame,(int(fut[i-1][0]),int(fut[i-1][1])),(int(fut[i][0]),int(fut[i][1])),(255,255,255),1,cv2.LINE_AA)

    cv2.rectangle(frame,(x1-2,y1-2),(x2+2,y2+2),tuple(c//2 for c in colour),1,cv2.LINE_AA)
    cv2.rectangle(frame,(x1,y1),(x2,y2),colour,2,cv2.LINE_AA)
    bl = min(18,(x2-x1)//4,(y2-y1)//4)
    for px,py,dx,dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(frame,(px,py),(px+dx*bl,py),colour,2,cv2.LINE_AA)
        cv2.line(frame,(px,py),(px,py+dy*bl),colour,2,cv2.LINE_AA)
    label = f"#{trk['id']} {cls_name} {speed:.1f}px/f"
    (tw,th),_ = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.45,1)
    lx,ly = x1, max(y1-8,12)
    cv2.rectangle(frame,(lx-2,ly-th-4),(lx+tw+4,ly+2),(10,5,25),cv2.FILLED)
    cv2.putText(frame,label,(lx,ly),cv2.FONT_HERSHEY_SIMPLEX,0.45,colour,1,cv2.LINE_AA)

def main():
    parser = argparse.ArgumentParser(description="VectraTrack CLI v2.0")
    parser.add_argument("--source",   type=str,   default="0")
    parser.add_argument("--tracker",  type=str,   default="custom", choices=["custom","bytetrack","botsort"])
    parser.add_argument("--conf",     type=float, default=0.35)
    parser.add_argument("--iou",      type=float, default=0.30)
    parser.add_argument("--max-age",  type=int,   default=30)
    parser.add_argument("--min-hits", type=int,   default=3)
    parser.add_argument("--trail",    action="store_true",  default=True)
    parser.add_argument("--no-trail", action="store_false", dest="trail")
    parser.add_argument("--predict",  action="store_true",  default=True)
    parser.add_argument("--no-predict",action="store_false",dest="predict")
    parser.add_argument("--trail-len",type=int,   default=30)
    parser.add_argument("--save",     type=str,   default=None)
    args = parser.parse_args()

    print("\n╔══════════════════════════════════╗")
    print("║   VECTRATRACK ML v2.0  ENGINE   ║")
    print("╚══════════════════════════════════╝")
    print(f"  Source : {args.source}  |  Tracker: {args.tracker.upper()}")
    print(f"  Conf   : {args.conf}    |  IoU    : {args.iou}")
    print(f"  Trail  : {args.trail}   |  Predict: {args.predict}\n")

    model  = YOLO("yolov8n.pt")
    source = 0 if args.source=="0" else args.source
    cap    = cv2.VideoCapture(source)
    if not cap.isOpened(): print("[ERROR] Cannot open source."); return

    w,h    = int(cap.get(3)), int(cap.get(4))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    writer = None
    if args.save:
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))
        print(f"  Saving : {args.save}")

    custom_tracker = None
    if args.tracker == "custom":
        custom_tracker = CustomSortTracker(max_age=args.max_age, min_hits=args.min_hits, iou_threshold=args.iou)

    total_frames = total_time = 0
    peak_objs = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        t0 = time.perf_counter()

        if custom_tracker is not None:
            results = model.predict(frame, conf=args.conf, verbose=False)
            boxes   = results[0].boxes
            dets = []
            if boxes is not None and len(boxes)>0:
                for box in boxes:
                    x1,y1,x2,y2 = map(float,box.xyxy[0])
                    dets.append([x1,y1,x2,y2,float(box.conf[0]),int(box.cls[0])])
            dets_np = np.array(dets) if dets else np.empty((0,6))
            tracks  = custom_tracker.update(dets_np)
        else:
            tyaml   = "bytetrack.yaml" if args.tracker=="bytetrack" else "botsort.yaml"
            results = model.track(frame, persist=True, conf=args.conf, tracker=tyaml, verbose=False)
            boxes   = results[0].boxes
            tracks  = []
            if boxes is not None and len(boxes)>0:
                for box in boxes:
                    if box.id is None: continue
                    x1,y1,x2,y2 = map(int,box.xyxy[0])
                    tracks.append({"id":int(box.id[0]),"bbox":[x1,y1,x2,y2],
                                   "cls":int(box.cls[0]),"conf":float(box.conf[0]),
                                   "speed":0.0,"history":[],"future":[],"age":1})

        elapsed      = time.perf_counter() - t0
        total_time  += elapsed
        total_frames+= 1
        peak_objs    = max(peak_objs, len(tracks))
        frame_fps    = 1.0/elapsed if elapsed>0 else 0

        for trk in tracks:
            draw_track(frame, trk, args.trail, args.predict, args.trail_len)
        cv2.putText(frame, f"VECTRATRACK | {args.tracker.upper()} | {len(tracks)} OBJ | {frame_fps:.1f} FPS",
                    (10,22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (168,85,247), 1, cv2.LINE_AA)

        cv2.imshow("VectraTrack ML", frame)
        if writer: writer.write(frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\nStopped by user."); break

    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()

    avg_fps = total_frames/total_time if total_time>0 else 0
    print(f"\n╔════════ SESSION STATS ════════╗")
    print(f"  Frames : {total_frames}  |  Time : {total_time:.2f}s")
    print(f"  Avg FPS: {avg_fps:.1f}   |  Peak : {peak_objs} objects")
    print(f"╚══════════════════════════════╝\n")

if __name__ == "__main__":
    main()
