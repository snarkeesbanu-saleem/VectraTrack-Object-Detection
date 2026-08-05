# VectraTrack 🚀

**VectraTrack** is a high-performance, browser-native computer vision suite designed for real-time object detection and multi-target motion tracking. Utilizing client-side hardware-accelerated deep neural networks paired with a customized mathematical tracking algorithm, VectraTrack parses sequential visual arrays to continuously isolate, lock, and project target vectors inside a stylized dark-neon HUD dashboard.

---

## 🌌 Key Architectural Features

*   **Real-time Neural Object Detection**: Integrates client-side **TensorFlow.js** utilizing optimized Mobilenet topologies. Performs fully localized inference frames on user camera inputs and media files.
*   **Sequential Bipartite Overlap Matching (SORT)**: Evaluates incoming raw visual detections against active model trajectories. Matches targets frame-over-frame using a greedy **Intersection over Union (IoU)** association grid.
*   **Linear Kinematics & Drift Projection**: Features a constant-velocity vector estimator. If a target is temporarily lost (e.g., behind structures or due to light changes), the engine extrapolates its path using historical displacement velocity.
*   **Virtual Radar Simulator Playground**: Includes a complete physics simulation environment featuring active movement profiles and distinct physical occlusion pillars. Perfect for testing tracker durability, path prediction, and target retrieval without using external assets.
*   **Cyber Surveillance HUD overlays**: Draws custom target bracket frames, direction arrows, path history trails, and unique ID labels over output feeds with adjustable glowing themes (*Cosmic Purple, Cyan-Grid, Cyberpunk Pink, Sci-Fi Amber*).
*   **Dynamic Data Plots & Log Telemetry**: Includes granular timeline line-charts tracking target densities, item class category breakdown bars, and a rolling log console streaming key system events.

---

## 🛠️ Technology Stack

*   **Frontend**: React 19 (TypeScript, Vite)
*   **Styles**: Tailwind CSS
*   **Animations**: Motion (Framer Motion)
*   **Inference Tensors**: TensorFlow.js Core
*   **Detections**: COCO-SSD (Lite MobileNetV2 Topology)
*   **Data Visualization**: Recharts

---

## 📁 Getting Started

### Prerequisites

*   **Node.js**: `v18+` or higher recommended.
*   **NPM / Bun / Yarn** package runner.

### Installation

1.  Clone the repository to your host environment.
2.  Install dependencies:
    ```bash
    npm install
    ```

### Dev Execution

To start the local development server mapping hot module reloads:

```bash
npm run dev
```

Open `http://localhost:3000` in your web browser.

### Compilation

To compile a minified standalone static bundle suitable for Vercel, GitHub Pages, or Netlify:

```bash
npm run build
```

The resulting assets will be compiled directly in the `/dist` output directory.

---

## ⚙️ Configurable Ground Parameters

*   **IoU Intersection Threshold**: Tighten or loosen matching requirements for multi-target overlaps between sequence frames.
*   **Confidence Ceiling**: Adjust model prediction filters (0.30 - 0.85) to eliminate environment noise.
*   **Lost Frame Retention (Survival)**: Configure how long (5 - 45 frames) the tracker uses dead-reckoning velocity estimations to predict occluded path drift.
*   **Taxonomy Filters**: Toggle filters on-the-fly to isolate specific classes (e.g., track vehicles or pedestrians while ignoring stationary objects).

---

## 🐍 ML Backend v2.0 (Python — `ml_backend/`)

A full Python-based Multi-Object Tracking pipeline has been added alongside the browser frontend.

### Features
| Feature | Details |
|---|---|
| **Custom SORT Tracker** | Built from scratch — Kalman Filter + Hungarian Algorithm bipartite matching (`custom_tracker.py`) |
| **Trajectory Prediction** | Neon past trails (60 frames) + dashed future projection (15 frames) via Kalman velocity states |
| **Streamlit Web App** | Upload video → choose algorithm → view analytics → download telemetry CSV (`app.py`) |
| **Benchmark Mode** | Run ByteTrack / BoT-SORT / Custom SORT on the same video and compare FPS |
| **CLI Tracker** | Real-time webcam / video tracking from the terminal (`tracker.py`) |
| **Telemetry Export** | Frame-level CSV: ID, class, bounding box, speed (px/frame), age |

### Algorithms
- **Kalman Filter** — 7D state vector `[cx, cy, s, r, vx, vy, vs]`, predict + update steps
- **Hungarian Algorithm** — `scipy.optimize.linear_sum_assignment` for optimal assignment
- **IoU metric** — detection-to-track association gate

### Quick Start
```bash
cd ml_backend
pip install -r requirements.txt

# Web App
streamlit run app.py

# CLI — Custom SORT on webcam
python tracker.py --tracker custom --trail --predict

# CLI — ByteTrack on video file, save output
python tracker.py --source video.mp4 --tracker bytetrack --save output.mp4
```

---

## 📄 License & Attribution

This project is licensed under the Apache-2.0 License. All visual components, state-estimation libraries, and modules are structured modularly to promote simple drops, audits, or scaling.
