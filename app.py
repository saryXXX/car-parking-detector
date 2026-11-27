"""
Streamlit Web Dashboard for Smart Parking Monitoring & Spatial Analytics
Interactive web interface displaying live occupancy metrics, video feeds,
slot-by-slot dwell times, lane obstruction alerts, bad parking detection,
and separate Near-Miss vs Actual Collision scenarios.
"""

import os
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import winsound

from src import model_cache, replay_cache
from src.pipeline import FramePipeline
from src.spatial_engine import SpatialEngine
from src.tracker import VehicleTracker
from src.video_io import FrameReader, read_frame_at
from src.visualizer import ParkingVisualizer


st.set_page_config(
    page_title="Parking Watch",
    page_icon="🅿️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

SIM_CONFIG = "configs/parking_slots.json"
UPLOAD_CONFIGS = [
    "configs/commercial_parking_slots.json",
    "configs/commercial_parking_zones.json",
    "configs/parking_slots.json",
]

EPISODES = [
    {
        "id": "e1", "badge": "E1", "kicker": "Simulation · Episode 1",
        "source": "Scénario 1 : Stationnement Régulier & Dwell Time",
        "title": "Regular Parking",
        "genres": ["Simulation", "Occupancy", "Dwell time"],
        "desc": "Car #101 crosses the entry tripwire, turns into Slot 01 and parks. "
                "Watch the occupancy counter and the dwell timer tick in real time.",
        "cars": ((101, 190, 190), (102, 350, 190)),
    },
    {
        "id": "e2", "badge": "E2", "kicker": "Simulation · Episode 2",
        "source": "Scénario 2 : Stationnement à Cheval sur la Ligne",
        "title": "Straddling the Line",
        "genres": ["Simulation", "Bad parking", "Two slots"],
        "desc": "Car #104 parks right across the line between Slot 04 and Slot 05. "
                "The spatial engine flags it as bad parking on both slots.",
        "cars": ((104, 270, 530),),
    },
    {
        "id": "e3", "badge": "E3", "kicker": "Simulation · Episode 3",
        "source": "Scénario 3 : Parking sur la Voie Principale",
        "title": "Blocking the Lane",
        "genres": ["Simulation", "Lane obstruction"],
        "desc": "Car #101 backs out of Slot 01 and stops at the edge of the main lane "
                "while Car #105 drives past.",
        "cars": ((101, 190, 295), (105, 330, 360)),
    },
    {
        "id": "e4", "badge": "E4", "kicker": "Simulation · Episode 4",
        "source": "Scénario 4 : Risque d'Accrochage Évité (Near-Miss)",
        "title": "Near Miss",
        "genres": ["Simulation", "Safety", "Impact avoided"],
        "desc": "Car #101 reverses into the lane without looking as Car #105 approaches. "
                "The engine catches the dangerous approach before impact.",
        "cars": ((101, 190, 345), (105, 95, 360)),
    },
    {
        "id": "e5", "badge": "E5", "kicker": "Simulation · Episode 5",
        "source": "Scénario 5 : Collision Réelle - Véhicules Bloqués",
        "title": "Collision",
        "genres": ["Simulation", "Safety", "Lane blocked"],
        "desc": "Car #101 reverses straight into the path of Car #105. Impact is detected, "
                "both vehicles stop and the lane is blocked.",
        "cars": ((101, 190, 330), (105, 150, 360)), "collided": True,
    },
    {
        "id": "r1", "badge": "4K", "kicker": "Real footage",
        "source": "📹 Vidéo Réelle 1 : Parking Commercial (Caméra Mobile GMC)",
        "title": "Commercial Lot",
        "genres": ["Real footage", "Moving camera", "GMC"],
        "desc": "Handheld 4K footage of a commercial lot. Global motion compensation keeps "
                "the slot map pinned to the ground while YOLO tracks every car.",
        "video": "14292633_3840_2160_30fps.mp4",
        # commercial_parking_slots.json was calibrated at the native 3840x2160 resolution
        # while the app always runs tracking/matching on a 1280x720 downscale -- its slot
        # polygons sit entirely outside that frame (3x too large), so every slot read as
        # permanently out-of-view/empty. commercial_parking_zones.json is the correct,
        # zone-aware calibration. Slots A1-A9 / B1-B9 are traced from the painted bay lines
        # (Hough), C1 is the white SUV by the kerb, P1-P54 are the other parked cars found by
        # tools/auto_slots_oblique.py.
        "config": "configs/commercial_parking_zones.json",
        # Camera looks across the lot: slot / contact tests use each car's ground band.
        "oblique": True,
        # Far rows are 30-40 px cars: YOLO at 640 sees ~38 cars per frame, at 1280 ~69.
        "imgsz": 1280,
    },
    {
        "id": "r2", "badge": "4K", "kicker": "Real footage",
        "source": "🚁 Vidéo Réelle 2 : Drone Zénithal 90° (Top-Down)",
        "title": "Drone's Eye View",
        "genres": ["Real footage", "Drone", "235 slots"],
        "desc": "A 90° top-down drone pass over a 235-slot lot. Occupancy for every "
                "single slot, seen from above.",
        "video": "4196207-uhd_3840_2160_24fps.mp4",
        # Hand-drawn drone_parking_lot.json was misaligned (polygons spanning 1.5-2 real
        # spaces, 141 of ~260 slots). This one is generated from the parked cars by
        # tools/auto_slots_from_detections.py.
        "config": "configs/drone_parking_lot_auto.json",
        # Top-down cars are invisible to COCO YOLO -> aerial-trained detector (falls back to
        # tiled COCO detection if the weights are missing, see src/aerial_detector.py).
        "aerial": True,
        "model": "models/geotrax_yolov8s_1920.pt",
    },
    {
        "id": "upload", "badge": "", "kicker": "Bring your own", "icon": "📁",
        "source": "📁 Importer un autre fichier vidéo (.mp4)",
        "title": "Your Footage",
        "genres": ["Your video", "MP4 · AVI · MOV"],
        "desc": "Upload your own parking video and run the full detection, tracking "
                "and spatial analytics pipeline on it.",
    },
    {
        "id": "webcam", "badge": "LIVE", "kicker": "Live", "icon": "📷",
        "source": "Webcam (Index 0)",
        "title": "Live Webcam",
        "genres": ["Live", "Camera 0"],
        "desc": "Point your webcam at a parking area and monitor it live.",
    },
]
EP_BY_ID = {ep["id"]: ep for ep in EPISODES}
ROWS = [
    ("Simulations", ["e1", "e2", "e3", "e4", "e5"]),
    ("Real Footage & Live", ["r1", "r2", "upload", "webcam"]),
]


def draw_sim_background(width=1280, height=720):
    bg = np.full((height, width, 3), (35, 35, 35), dtype=np.uint8)
    cv2.rectangle(bg, (0, 300), (width, 420), (50, 50, 50), -1)
    for x in range(0, width, 60):
        cv2.line(bg, (x, 360), (x + 30, 360), (0, 255, 255), 2)
    cv2.line(bg, (140, 300), (140, 420), (255, 255, 0), 3)
    cv2.putText(bg, "TRIPWIRE ENTREE", (145, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 0), 1)
    return bg


def episode_frame(ep_id):
    ep = EP_BY_ID[ep_id]
    if "cars" in ep:
        engine = SpatialEngine(config_path=SIM_CONFIG)
        visualizer = ParkingVisualizer(slot_definitions=engine.slots)
        bg = draw_sim_background()
        color = (0, 0, 255) if ep.get("collided") else (80, 80, 200)
        tracked = []
        for car_id, x, y in ep["cars"]:
            bbox = (x - 40, y - 55, x + 40, y + 55)
            tracked.append({"id": car_id, "bbox": bbox, "conf": 0.95, "cls_name": "car"})
            cv2.rectangle(bg, bbox[:2], bbox[2:], color, -1)
        analytics = engine.update(tracked, current_time=time.time())
        return visualizer.render(bg, analytics, tracked, fps=30.0)
    if ep.get("video") and os.path.exists(ep["video"]):
        return read_frame_at(ep["video"], 90)
    return None


# Thumbnails are written once to static/thumbs/ and served by Streamlit's static file server
# (server.enableStaticServing). Before, each one cost a 4K seek (~1.2-3 s) after every server
# restart, and all of them were re-sent as inline base64 on every rerun.
THUMB_DIR = Path("static/thumbs")
THUMB_WIDTHS = (480, 1280)


def _thumb_sig(ep_id):
    src = EP_BY_ID[ep_id].get("video") or "src/visualizer.py"  # sim thumbnails follow the renderer
    return f"{int(os.path.getmtime(src))}" if os.path.exists(src) else "0"


@st.cache_data(show_spinner=False)
def episode_image(ep_id, width):
    """Relative URL of the episode thumbnail at `width`, generated on first use."""
    sig = _thumb_sig(ep_id)
    paths = {w: THUMB_DIR / f"{ep_id}_{w}_{sig}.jpg" for w in THUMB_WIDTHS}
    if not paths[width].exists():
        frame = episode_frame(ep_id)
        if frame is None:
            return None
        THUMB_DIR.mkdir(parents=True, exist_ok=True)
        h, w = frame.shape[:2]
        for tw, path in paths.items():  # one decode serves every size
            small = cv2.resize(frame, (tw, int(h * tw / w)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(path), small, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return f"./app/static/thumbs/{paths[width].name}"


st.session_state.setdefault("selected", "e5")
st.session_state.setdefault("playing", False)


def select_episode(ep_id):
    st.session_state.selected = ep_id
    st.session_state.playing = False


def start_playing():
    st.session_state.playing = True


def stop_playing():
    st.session_state.playing = False


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;500;600;700&display=swap');
:root { --pw-red:#e50914; --pw-bg:#141414; --pw-surface:#1c1c1c; --pw-line:#2b2b2b; --pw-text:#e5e5e5; --pw-muted:#a3a3a3; }
.stApp { background: var(--pw-bg); }
.stApp, .stApp button, .stApp input { font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stMainBlockContainer"] { max-width: none; padding: 1rem 3.5rem 4rem; }
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="stExpandSidebarButton"] { display: none; }

.pw-logo { font-family: 'Bebas Neue', Impact, sans-serif; color: var(--pw-red); font-size: 2.6rem; letter-spacing: .03em; line-height: 1; }

.st-key-hero { min-height: min(66vh, 620px); padding: 3rem 3rem 3.25rem; border-radius: 8px;
  background-size: cover; background-position: center; justify-content: flex-end; gap: .75rem; }
.pw-kicker { display: flex; align-items: center; gap: .6rem; color: var(--pw-muted); font-size: .8rem; font-weight: 700; letter-spacing: .25em; text-transform: uppercase; }
.pw-n { font-family: 'Bebas Neue', Impact, sans-serif; color: var(--pw-red); font-size: 1.9rem; letter-spacing: 0; line-height: 1; }
.pw-title { font-family: 'Bebas Neue', Impact, sans-serif; color: #fff; font-size: clamp(3rem, 7vw, 6.5rem); line-height: .9; margin: .4rem 0 .6rem; text-shadow: 0 2px 24px rgba(0,0,0,.6); }
.pw-meta { color: var(--pw-text); font-size: 1rem; font-weight: 500; }
.pw-tag { color: #46d369; font-weight: 700; }
.pw-dot { color: #6b6b6b; margin: 0 .45rem; }
.pw-desc { color: var(--pw-text); font-size: 1.15rem; line-height: 1.5; max-width: 36rem; margin-top: .8rem; text-shadow: 0 1px 8px rgba(0,0,0,.8); }
.st-key-hero [data-testid="stFileUploader"], .st-key-hero [data-testid="stSelectbox"] { max-width: 30rem; }

.st-key-hero_play { margin-top: .9rem; }
.st-key-hero_play button { background: #fff; border: none; padding: .65rem 2.2rem; border-radius: 4px; }
.st-key-hero_play button:hover, .st-key-hero_play button:focus:not(:active) { background: rgba(255,255,255,.75); border: none; }
.st-key-hero_play button p { color: #000; font-size: 1.15rem; font-weight: 700; }
.st-key-hero_play button:disabled { background: rgba(109,109,110,.7); }
.st-key-hero_play button:disabled p { color: #d0d0d0; }

.pw-row-title { color: var(--pw-text); font-size: 1.4rem; font-weight: 700; margin: 2.25rem 0 .2rem; }
[class*="st-key-card_"] { position: relative; gap: 0; transition: transform .25s ease; }
[class*="st-key-card_"]:hover { transform: scale(1.07); z-index: 10; }
[class*="st-key-open_"] { position: absolute !important; inset: 0; z-index: 5; }
[class*="st-key-open_"] .stButton, [class*="st-key-open_"] button { width: 100%; height: 100%; }
[class*="st-key-open_"] button { opacity: 0; cursor: pointer; }
.pw-card { border-radius: 6px; overflow: hidden; background: var(--pw-surface); transition: box-shadow .25s ease; }
[class*="st-key-card_"]:hover .pw-card { box-shadow: 0 14px 30px rgba(0,0,0,.75); }
.pw-card.is-selected { outline: 2px solid var(--pw-red); outline-offset: -2px; }
.pw-thumb { position: relative; aspect-ratio: 16 / 9; background: #222; }
.pw-thumb img { width: 100%; height: 100%; object-fit: cover; display: block; }
.pw-thumb::after { content: ""; position: absolute; inset: 0; background: linear-gradient(0deg, rgba(0,0,0,.6), rgba(0,0,0,0) 45%); }
.pw-thumb-fallback { width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; font-size: 3rem;
  background: radial-gradient(circle at 70% 30%, #4a0b0f, #1c1c1c 70%); }
.pw-ep { position: absolute; left: .55rem; bottom: .3rem; z-index: 1; font-family: 'Bebas Neue', Impact, sans-serif; font-size: 1.6rem; color: #fff; line-height: 1; }
.pw-card-title { color: #fff; font-weight: 700; font-size: .95rem; padding: .6rem .75rem 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.pw-card-meta { color: var(--pw-muted); font-size: .8rem; padding: .15rem .75rem .75rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

.st-key-back button { background: transparent; border: 1px solid #5a5a5a; }
.st-key-back button:hover { border-color: #fff; color: #fff; }
.pw-player-title { font-family: 'Bebas Neue', Impact, sans-serif; font-size: 2.6rem; color: #fff; line-height: 1; }
.pw-player-title span { color: var(--pw-red); margin-right: .5rem; }
[data-testid="stMetric"] { background: var(--pw-surface); border: 1px solid var(--pw-line); border-radius: 6px; padding: .9rem 1.1rem; }
[data-testid="stMetricLabel"] p { color: var(--pw-muted); font-size: .78rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; }
[data-testid="stMetricValue"] { color: #fff; font-weight: 700; }
[data-testid="stImage"] img { border-radius: 6px; }
.pw-side-title { color: var(--pw-text); font-size: 1.05rem; font-weight: 700; margin-bottom: .4rem; }

@media (max-width: 768px) {
  [data-testid="stMainBlockContainer"] { padding: 1rem 1rem 3rem; }
  .st-key-hero { padding: 1.5rem; min-height: 60vh; }
}
</style>
""", unsafe_allow_html=True)


model_map = {
    "yolov8n.pt (Baseline COCO)": "yolov8n.pt",
    # Pixel-mask instance segmentation instead of plain bboxes -- more precise slot-overlap
    # computation on oblique/angled cameras. Auto-downloads from Ultralytics on first use,
    # same as yolov8n.pt above (both are free, local-inference, no paid API involved).
    "yolov8n-seg.pt (Segmentation - pixel masks)": "yolov8n-seg.pt",
}
if os.path.exists("models"):
    for pt_file in sorted(Path("models").glob("*.pt")):
        fname = pt_file.name
        if fname.startswith("geotrax"):
            label = f"{fname} (Aerial · vehicles, CC-BY-4.0)"
        elif fname.startswith("visdrone"):
            label = f"{fname} (Aerial · VisDrone, AGPL)"
        elif fname.startswith("yolov8n_parking"):
            # These checkpoints collapsed in training (conf 1.00 on trees/grass); kept for reference.
            label = f"{fname} (⚠ broken fine-tune)"
        elif "v1" in fname:
            label = f"{fname} (V1 · drone fine-tune)"
        elif "v2" in fname:
            label = f"{fname} (V2 · Colab fine-tune)"
        else:
            label = f"{fname} ({pt_file.stat().st_size / (1024 * 1024):.1f} MB)"
        model_map[label] = str(pt_file).replace("\\", "/")

nav_logo, nav_settings = st.columns([5, 1], vertical_alignment="center")
with nav_logo:
    st.markdown('<div class="pw-logo">Parking Watch</div>', unsafe_allow_html=True)
with nav_settings:
    with st.popover("⚙  Model settings", width="stretch"):
        selected_label = st.selectbox("Model version", list(model_map.keys()),
                                      index=1 if len(model_map) > 1 else 0, key="model_label")
        model_choice = model_map[selected_label]
        st.caption(f"Active file: `{model_choice}`")
        conf_thresh = st.slider("YOLO confidence threshold", 0.10, 0.90, 0.25, 0.05, key="conf_thresh")
        slot_weights_path = "runs/classify/train/weights/best.pt"
        slot_weights_exist = os.path.exists(slot_weights_path)
        occupancy_mode_label = st.selectbox(
            "Occupation : Règles / CNN / Fusion",
            ["Règles", "CNN", "Fusion"],
            index=0, key="occupancy_mode_label",
            disabled=not slot_weights_exist,
            help=None if slot_weights_exist else
                 "Entraînez d'abord le classifieur de places (tools/build_slot_dataset.py "
                 "puis `yolo classify train`) pour activer ce mode.",
        )
        occupancy_mode = {"Règles": "rule", "CNN": "cnn", "Fusion": "fusion"}[occupancy_mode_label]
        collision_weights_path = "models/collision_lgbm.txt"
        collision_weights_exist = os.path.exists(collision_weights_path)
        collision_mode_label = st.selectbox(
            "Collisions : Règles / ML",
            ["Règles", "ML"],
            index=0, key="collision_mode_label",
            disabled=not collision_weights_exist,
            help=None if collision_weights_exist else
                 "Entraînez d'abord le modèle (tools/simulate_trajectories.py puis "
                 "tools/train_collision_model.py) pour activer ce mode.",
        )
        collision_mode = {"Règles": "rule", "ML": "ml"}[collision_mode_label]

# Load the default detectors and run one dummy inference in the background as soon as the app
# opens: the cold start (CUDA init + weights, ~15-20 s) used to land on the first Play click.
# Runs once per server process; later reruns are no-ops.
_warm_specs = [(model_choice, "track", 1280)] if os.path.exists(model_choice) else []
_warm_specs += [(ep["model"], "detect", 1280) for ep in EPISODES if ep.get("aerial") and os.path.exists(ep["model"])]
model_cache.start_warmup(_warm_specs)


def render_card(ep):
    img = episode_image(ep["id"], 480)
    thumb = f'<img src="{img}" alt="">' if img else f'<div class="pw-thumb-fallback">{ep["icon"]}</div>'
    badge = f'<span class="pw-ep">{ep["badge"]}</span>' if ep["badge"] else ""
    selected = " is-selected" if ep["id"] == st.session_state.selected else ""
    meta = " · ".join(ep["genres"][1:]) or ep["genres"][0]
    with st.container(key=f"card_{ep['id']}"):
        st.markdown(
            f'<div class="pw-card{selected}"><div class="pw-thumb">{thumb}{badge}</div>'
            f'<div class="pw-card-title">{ep["title"]}</div><div class="pw-card-meta">{meta}</div></div>',
            unsafe_allow_html=True,
        )
        st.button(f"Open {ep['title']}", key=f"open_{ep['id']}", on_click=select_episode, args=(ep["id"],))


def render_upload_controls():
    uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"], key="upload_file")
    if uploaded is not None:
        signature = (uploaded.name, uploaded.size)
        if st.session_state.get("upload_signature") != signature:
            os.makedirs("data/temp", exist_ok=True)
            path = os.path.join("data/temp", os.path.basename(uploaded.name))
            with open(path, "wb") as f:
                f.write(uploaded.getbuffer())
            st.session_state.upload_path = path
            st.session_state.upload_signature = signature
    current = st.session_state.get("upload_config", UPLOAD_CONFIGS[0])
    st.session_state.upload_config = st.selectbox(
        "Slot layout", UPLOAD_CONFIGS, index=UPLOAD_CONFIGS.index(current), key="upload_config_widget"
    )


def render_webcam_controls():
    st.session_state.webcam_url = st.text_input(
        "Camera source (leave blank for local webcam index 0)",
        value=st.session_state.get("webcam_url", ""),
        key="webcam_url_widget",
        help="Local webcam: leave blank. IP camera / NVR: rtsp://user:pass@host:554/stream1 . "
             "HTTP MJPEG: http://host/video . Anything OpenCV's VideoCapture accepts works here.",
    )
    current = st.session_state.get("webcam_config", UPLOAD_CONFIGS[0])
    st.session_state.webcam_config = st.selectbox(
        "Slot layout (calibrate with src/calibrate.py --source <same URL> first)",
        UPLOAD_CONFIGS, index=UPLOAD_CONFIGS.index(current), key="webcam_config_widget",
    )


def render_browse(ep):
    hero_img = episode_image(ep["id"], 1280)
    backdrop = f'url("{hero_img}")' if hero_img else "radial-gradient(circle at 75% 40%, #5c0a0f 0%, #141414 70%)"
    st.markdown(
        "<style>.st-key-hero { background-image: "
        "linear-gradient(90deg, rgba(20,20,20,.95) 0%, rgba(20,20,20,.7) 38%, rgba(20,20,20,0) 72%), "
        f"linear-gradient(0deg, #141414 0%, rgba(20,20,20,0) 35%), {backdrop}; }}</style>",
        unsafe_allow_html=True,
    )
    meta = '<span class="pw-dot">•</span>'.join(
        [f'<span class="pw-tag">{ep["genres"][0]}</span>'] + ep["genres"][1:]
    )
    with st.container(key="hero"):
        st.markdown(
            f'<div class="pw-kicker"><span class="pw-n">P</span>{ep["kicker"]}</div>'
            f'<div class="pw-title">{ep["title"]}</div>'
            f'<div class="pw-meta">{meta}</div>'
            f'<div class="pw-desc">{ep["desc"]}</div>',
            unsafe_allow_html=True,
        )
        if ep["id"] == "upload":
            render_upload_controls()
        elif ep["id"] == "webcam":
            render_webcam_controls()
        can_play = ep["id"] != "upload" or bool(st.session_state.get("upload_path"))
        st.button("▶  Play", key="hero_play", on_click=start_playing, disabled=not can_play)

    for row_title, ids in ROWS:
        st.markdown(f'<div class="pw-row-title">{row_title}</div>', unsafe_allow_html=True)
        for col, ep_id in zip(st.columns(5, gap="small"), ids):
            with col:
                render_card(EP_BY_ID[ep_id])


def render_player(ep):
    # The playback loop never ends the script run, so leftovers from the browse screen would stay visible.
    st.markdown('<style>[data-stale="true"] { display: none; }</style>', unsafe_allow_html=True)
    back_col, title_col = st.columns([1, 6], vertical_alignment="center")
    back_col.button("←  Back to browse", key="back", on_click=stop_playing, width="stretch")
    title_col.markdown(
        f'<div class="pw-player-title"><span>{ep["badge"] or "▶"}</span>{ep["title"]}</div>',
        unsafe_allow_html=True,
    )
    metrics = [col.empty() for col in st.columns(5, gap="small")]
    col_video, col_side = st.columns([7, 3], gap="medium")
    with col_video:
        status = st.empty()   # analysis progress / replay info
        video = st.empty()
        seek = st.empty()     # "start from" slider, replays only
        alerts = st.empty()
    with col_side:
        st.markdown('<div class="pw-side-title">Slot status</div>', unsafe_allow_html=True)
        table = st.empty()
    return metrics, video, table, alerts, status, seek


episode = EP_BY_ID[st.session_state.selected]
# A single slot for the page body, so the browse screen is fully replaced while the player loop runs.
with st.empty().container():
    if st.session_state.playing:
        (
            (metric_free, metric_occ, metric_rate, metric_entries, metric_incidents),
            video_placeholder, table_placeholder, alert_placeholder, status_placeholder, seek_placeholder,
        ) = render_player(episode)
    else:
        render_browse(episode)

source_type = episode["source"]
video_path = episode.get("video")
config_file = episode.get("config", SIM_CONFIG)
if episode["id"] == "upload":
    video_path = st.session_state.get("upload_path")
    config_file = st.session_state.get("upload_config", UPLOAD_CONFIGS[0])
elif episode["id"] == "webcam":
    config_file = st.session_state.get("webcam_config", UPLOAD_CONFIGS[0])


JPEG_QUALITY = 85
PANEL_REFRESH_S = 0.5  # metrics + slot table: ~2 Hz instead of every frame


def encode_jpeg(bgr: np.ndarray) -> bytes:
    # Encoded once here; st.image passes JPEG bytes straight through (no re-encode), and the
    # same bytes are what the replay cache stores.
    return cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])[1].tobytes()


class PlayerUI:
    """Pushes frames to the page every time, but the metrics and the slot table (a full
    dataframe re-sent to the browser) only ~2x per second, and alerts only when they change."""

    def __init__(self, slots):
        self.slots = slots  # [(slot_id, name), ...]
        self._last_panel = 0.0
        self._last_alert = None

    def show(self, jpeg: bytes, rec: dict, force: bool = False):
        video_placeholder.image(jpeg, width="stretch")
        now = time.perf_counter()
        if force or now - self._last_panel >= PANEL_REFRESH_S:
            self._last_panel = now
            self._panel(rec)
        alert = self._alert(rec)
        if alert != self._last_alert:
            self._last_alert = alert
            if alert is None:
                alert_placeholder.empty()
            elif alert[0] == "error":
                alert_placeholder.error(alert[1])
            else:
                alert_placeholder.warning(alert[1])

    def _panel(self, rec):
        incidents = len(rec["obstructions"]) + len(rec["straddles"]) + len(rec["collisions"])
        metric_free.metric("Places Libres", f"{rec['free']} / {rec['total']}")
        metric_occ.metric("Places Occupées", f"{rec['occ']}")
        metric_rate.metric("Taux d'Occupation", f"{rec['rate']:.1f} %")
        metric_entries.metric("Véhicules Entrés", f"{rec['entries']}")
        metric_incidents.metric("Incidents Détectés", f"{incidents}")

        occupied = set(rec["occupied"])
        rows = []
        for sid, name in self.slots:
            if sid in occupied:
                car_id, elapsed = rec["dwell"].get(str(sid), [None, 0.0])
                mins, secs = divmod(int(elapsed), 60)
                rows.append({"Place": name, "État": "🔴 Occupée",
                             "Véhicule": f"Car #{car_id}" if car_id else "-",
                             "Temps Stationné": f"{mins:02d}m {secs:02d}s"})
            else:
                rows.append({"Place": name, "État": "🟢 Libre", "Véhicule": "-", "Temps Stationné": "-"})
        table_placeholder.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    @staticmethod
    def _alert(rec):
        errors, warnings = [], []
        for col in rec["collisions"]:  # COLLISION (highest priority)
            if col["type"] == "COLLISION":
                errors.append(f"🚨💥 **COLLISION CONFIRMÉE** entre **Car #{col['car1_id']}** x **Car #{col['car2_id']}** !")
            else:
                warnings.append(f"⚠️⚡ **RISQUE D'ACCROCHAGE DÉTECTÉ** : **Car #{col['car1_id']}** ↔ **Car #{col['car2_id']}** — Rapprochement d'urgence détecté !")
        for stra in rec["straddles"]:
            slots_str = ", ".join(f"Slot {sid:02d}" for sid in stra["slot_ids"])
            errors.append(f"⛔ **MAUVAIS STATIONNEMENT** : **Car #{stra['car_id']}** à cheval sur {slots_str} !")
        for alt in rec["obstructions"]:
            warnings.append(f"⚠️ **VOIE BLOQUÉE** : **Car #{alt['car_id']}** immobilisé ({alt['duration']:.1f}s)")
        if errors:
            return ("error", " | ".join(errors))
        if warnings:
            return ("warning", " | ".join(warnings[:2]))  # max 2 warnings
        return None


def build_engine():
    slot_classifier = None
    if occupancy_mode != "rule" and slot_weights_exist:
        from src.slot_classifier import SlotClassifier
        slot_classifier = SlotClassifier(slot_weights_path, every_n_frames=3)

    # Built without a collision_model first so we can read the engine's resolved
    # px_per_metre (explicit > calibrated via src/calibrate.py --ref-metres > 75px/m
    # fallback) and hand the ML model the SAME scale, instead of each guessing separately.
    engine = SpatialEngine(config_path=config_file, occupancy_mode=occupancy_mode,
                           slot_classifier=slot_classifier, collision_mode=collision_mode,
                           top_down=episode.get("aerial", False),
                           oblique=episode.get("oblique", False))
    if collision_mode == "ml" and collision_weights_exist:
        from src.collision_model import CollisionModel
        engine.collision_model = CollisionModel(collision_weights_path,
                                                pixel_to_metre_scale=1.0 / engine.px_per_metre)
    return engine


def wait_for_models():
    if not model_cache.warmup_ready():
        with status_placeholder, st.spinner("Warming up the detector on the GPU (first run after start-up only)…"):
            model_cache.wait_warmup()


def run_simulation():
    engine = build_engine()
    visualizer = ParkingVisualizer(slot_definitions=engine.slots)
    ui = PlayerUI([(s["id"], s["name"]) for s in engine.slots])
    frame_idx = 0
    width, height = 1280, 720

    # Véhicules pour les différents scénarios
    car_101 = {"id": 101, "x": 50.0, "y": 360.0}
    car_102 = {"id": 102, "x": 350.0, "y": 190.0}
    car_104 = {"id": 104, "x": 60.0, "y": 360.0}
    car_105 = {"id": 105, "x": 50.0, "y": 360.0}
    collision_occurred = False
    collision_sound_played = False

    while True:
        curr_time = time.time()
        # Rendu du fond de plan parking 2D propre (Voie de circulation + Marquage)
        bg = draw_sim_background(width, height)

        tracked = []

        if "Scénario 1" in source_type:
            # Stationnement régulier : Car 101 avance puis tourne dans Slot 01 (190, 190)
            if frame_idx < 40:
                car_101["x"] += (190.0 - 50.0) / 40.0
                car_101["y"] = 360.0
            elif frame_idx < 80:
                car_101["x"] = 190.0
                car_101["y"] -= (360.0 - 190.0) / 40.0
            else:
                car_101["x"] = 190.0
                car_101["y"] = 190.0
            active_cars = [car_101, car_102]

        elif "Scénario 2" in source_type:
            # À cheval sur la ligne : Car 104 s'arrête à x=270, pile entre Slot 04 et Slot 05
            if frame_idx < 45:
                car_104["x"] += (270.0 - 60.0) / 45.0
                car_104["y"] = 360.0
            elif frame_idx < 85:
                car_104["x"] = 270.0
                car_104["y"] += (530.0 - 360.0) / 40.0
            else:
                car_104["x"] = 270.0
                car_104["y"] = 530.0
            active_cars = [car_104]

        elif "Scénario 3" in source_type:
            # Near-Miss : Car 101 recule de Slot 01 vers l'allée, aperçoit Car 105 et freine à frame 45 (y=295)
            if frame_idx == 0:
                car_101["x"], car_101["y"] = 190.0, 190.0
            car_105["x"] += 5.0
            if 15 <= frame_idx < 45:
                car_101["y"] += 3.5
            elif frame_idx >= 45:
                car_101["y"] = 295.0  # Freinage d'urgence !
            active_cars = [car_101, car_105]

        elif "Scénario 4" in source_type:
            # Near-Miss : Car 101 recule sans regarder jusqu'au centre de la voie (y=360)
            if frame_idx == 0:
                car_101["x"], car_101["y"] = 190.0, 190.0
            if not collision_occurred:
                car_105["x"] += 5.0
                if frame_idx >= 15 and car_101["y"] < 360.0:
                    car_101["y"] += 4.5
                if abs(car_105["x"] - car_101["x"]) < 65 and abs(car_105["y"] - car_101["y"]) < 45:
                    collision_occurred = True
            active_cars = [car_101, car_105]

        elif "Scénario 5" in source_type:
            # REAL COLLISION : Car 101 reverses from parking INTO lane, Car 105 drives through lane
            if frame_idx == 0:
                car_101["x"], car_101["y"] = 190.0, 190.0
                car_105["x"] = 50.0
                car_105["y"] = 360.0  # Car 105 in the lane

            if not collision_occurred:
                # Car 105 drives RIGHT through the lane (x-axis along y=360)
                car_105["x"] += 3.5

                # Car 101 GRADUALLY reverses INTO the lane (y → 360)
                if frame_idx >= 15:
                    car_101["y"] += 2.5

                # Distance on x-axis (when both reach same y=360)
                x_dist = abs(car_105["x"] - car_101["x"])
                y_dist = abs(car_105["y"] - car_101["y"])

                # COLLISION: when Car 101 reaches lane AND cars are getting close
                # Trigger EARLY before overlap: y_dist < 60 AND x_dist < 180
                if frame_idx >= 30 and y_dist < 60 and x_dist < 180:
                    collision_occurred = True

            # After collision: both cars stay stopped

            active_cars = [car_101, car_105]

        for c in active_cars:
            cx, cy = int(c["x"]), int(c["y"])
            w, h = 80, 110
            bbox = (cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2)
            tracked.append({"id": c["id"], "bbox": bbox, "conf": 0.95, "cls_name": "car"})

            # Red color when collision occurs
            if collision_occurred:
                color = (0, 0, 255)  # Red for collision
            else:
                color = (80, 80, 200)  # Normal blue

            cv2.rectangle(bg, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, -1)

        analytics = engine.update(tracked, current_time=curr_time)
        vis = visualizer.render(bg, analytics, tracked, fps=30.0)

        if "Scénario 5" in source_type:
            x_dist = abs(car_105["x"] - car_101["x"])
            y_dist = abs(car_105["y"] - car_101["y"])
            impact_pt = (int((car_101["x"] + car_105["x"]) / 2), 360)

            if collision_occurred:
                # FULL COLLISION: Red pulse + alert
                pulse = int((frame_idx % 8) * 5)
                cv2.circle(vis, impact_pt, 40 + pulse, (0, 0, 255), 3, cv2.LINE_AA)
                cv2.circle(vis, impact_pt, 60 + pulse, (0, 0, 255), 2, cv2.LINE_AA)

                # Play collision sound (only once)
                if not collision_sound_played:
                    try:
                        winsound.Beep(1000, 300)  # 1000Hz beep for 300ms
                        collision_sound_played = True
                    except:
                        pass

                # Blinking collision warning box - LARGE with solid background
                if int((frame_idx % 10) / 5) == 0:
                    # Solid red background box
                    cv2.rectangle(vis, (50, 280), (950, 380), (0, 0, 200), -1)
                    # White border
                    cv2.rectangle(vis, (50, 280), (950, 380), (255, 255, 255), 3)
                    # Text
                    cv2.putText(vis, "COLLISION DETECTEE - VOIE BLOQUEE",
                               (70, 345), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)

            elif y_dist < 40 and 50 < x_dist < 130:
                # NEAR-MISS: Orange warning pulse (cars getting close)
                pulse = int((frame_idx % 6) * 4)
                cv2.circle(vis, impact_pt, 30 + pulse, (0, 165, 255), 3, cv2.LINE_AA)
                cv2.circle(vis, impact_pt, 60 + pulse, (0, 165, 255), 2, cv2.LINE_AA)
                cv2.putText(vis, "RAPPROCHEMENT DANGEREUX", (impact_pt[0] - 180, impact_pt[1] - 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)

        elif collision_occurred:
            # Other scenarios: just show pulse ring
            pulse = int((frame_idx % 8) * 5)
            impact_pt = (int((car_101["x"] + car_105["x"]) / 2), 360)
            cv2.circle(vis, impact_pt, 40 + pulse, (0, 0, 255), 3, cv2.LINE_AA)

        ui.show(encode_jpeg(vis), replay_cache.ui_record(analytics))
        frame_idx += 1
        time.sleep(0.03)  # simulations render in ~ms: pace them to ~25 fps


def run_live():
    live_source = st.session_state.get("webcam_url", "").strip() or 0
    cap = cv2.VideoCapture(live_source)
    if not cap.isOpened():
        st.error(f"Impossible d'ouvrir la source live '{live_source}'.")
        return
    wait_for_models()
    engine = build_engine()
    tracker = VehicleTracker(model_name=model_choice, conf_threshold=conf_thresh)
    pipe = FramePipeline(engine, tracker, ParkingVisualizer(slot_definitions=engine.slots))
    ui = PlayerUI([(s["id"], s["name"]) for s in engine.slots])
    status_placeholder.caption(f"🔴 Live · {live_source}")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                # Live feed dropped a frame / connection hiccup: reconnect instead of ending.
                cap.release()
                time.sleep(1.0)
                cap = cv2.VideoCapture(live_source)
                if not cap.isOpened():
                    st.warning(f"Flux live '{live_source}' indisponible, nouvelle tentative...")
                continue
            if frame.shape[1] > 1280:
                frame = cv2.resize(frame, (1280, 720))
            vis, analytics, _ = pipe.process(frame, time.time())
            ui.show(encode_jpeg(vis), replay_cache.ui_record(analytics))
    finally:
        cap.release()
        pipe.close()


def run_video(path):
    ep_model = episode.get("model")
    if ep_model and os.path.exists(ep_model):
        st.caption(f"Detector for this video: `{ep_model}` (aerial)")
    else:
        ep_model = model_choice
    imgsz = episode.get("imgsz", 640)
    key = replay_cache.make_key(
        path, config_file, ep_model,
        extra_files=[slot_weights_path if occupancy_mode != "rule" else None,
                     collision_weights_path if collision_mode == "ml" else None],
        conf=conf_thresh, imgsz=imgsz, aerial=episode.get("aerial", False),
        oblique=episode.get("oblique", False), occupancy=occupancy_mode, collision=collision_mode,
    )
    replay = replay_cache.load(key)
    if replay is not None:
        play_replay(replay, key)
    else:
        analyse_and_record(path, key, ep_model, imgsz)


def play_replay(replay, key):
    """Stream a recorded analysis at the clip's own frame rate (frames are dropped, not
    slowed down, if the browser can't keep up)."""
    n, fps = replay.frame_count, replay.fps
    duration = n / fps
    start_s = seek_placeholder.slider("Start from (s)", 0.0, max(0.1, round(duration, 1)), 0.0, 0.5,
                                      key=f"seek_{episode['id']}")
    status_placeholder.caption(f"⚡ Cached analysis · replaying at {fps:.0f} fps · {n} frames "
                               "(settings or code changes trigger a fresh analysis)")
    ui = PlayerUI([tuple(s) for s in replay.meta["slots"]])
    i = min(n - 1, int(start_s * fps))
    t0 = time.perf_counter() - i / fps
    try:
        while i < n:
            lag = time.perf_counter() - (t0 + i / fps)
            if lag < 0:
                time.sleep(-lag)
            elif lag > 2 / fps:  # running late: jump to the frame that is due now
                i = min(n - 1, int((time.perf_counter() - t0) * fps))
            ui.show(replay.frame(i), replay.records[i], force=(i == n - 1))
            i += 1
        st.info("Flux vidéo terminé.")
    finally:
        replay.close()


def analyse_and_record(path, key, ep_model, imgsz):
    reader = FrameReader(path)
    if not reader.isOpened():
        reader.release()
        st.warning("Vidéo introuvable ou illisible.")
        return
    wait_for_models()
    engine = build_engine()
    tracker = VehicleTracker(model_name=ep_model, conf_threshold=conf_thresh,
                             aerial=episode.get("aerial", False), fps=reader.fps, imgsz=imgsz)
    pipe = FramePipeline(engine, tracker, ParkingVisualizer(slot_definitions=engine.slots))
    slots = [(s["id"], s["name"]) for s in engine.slots]
    ui = PlayerUI(slots)
    writer = replay_cache.ReplayWriter(key, meta={"fps": reader.fps, "slots": slots,
                                                  "video": os.path.basename(path)})
    total = reader.frame_count
    # Recorded file: use the video clock. Processing runs slower than real time,
    # so wall-clock time would shrink every speed (and stretch dwell times).
    video_t0 = time.time()
    started, last_status, done = time.perf_counter(), 0.0, 0
    try:
        while True:
            ok, frame, pos_ms = reader.read()
            if not ok:
                break
            vis, analytics, _ = pipe.process(frame, video_t0 + pos_ms / 1000.0, fps_display=reader.fps)
            jpeg, rec = encode_jpeg(vis), replay_cache.ui_record(analytics)
            writer.add(jpeg, rec)
            done += 1
            ui.show(jpeg, rec)
            now = time.perf_counter()
            if now - last_status >= PANEL_REFRESH_S:
                last_status = now
                speed = done / (now - started)
                text = (f"Analysing {done}/{total or '?'} frames · {speed:.1f} fps — "
                        f"the next Play replays this clip at {reader.fps:.0f} fps")
                status_placeholder.progress(min(1.0, done / total) if total else 0.0, text=text)
        if done:
            ui.show(jpeg, rec, force=True)
            writer.commit()
            status_placeholder.success(f"Analysis saved ({done} frames). Press Play again to replay it at "
                                       f"{reader.fps:.0f} fps.")
        st.info("Flux vidéo terminé.")
    finally:
        writer.abort()  # no-op after commit(); drops the partial recording if playback was interrupted
        reader.release()
        pipe.close()


def run_dashboard():
    if not os.path.exists(config_file):
        st.error(f"Fichier de configuration '{config_file}' introuvable.")
        return
    if "Scénario" in source_type:
        run_simulation()
    elif "Webcam" in source_type:
        run_live()
    elif not video_path or not os.path.exists(video_path):
        st.warning("Vidéo introuvable ou non sélectionnée.")
    else:
        run_video(video_path)


if st.session_state.playing:
    run_dashboard()
