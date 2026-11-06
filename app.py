"""
Streamlit Web Dashboard for Smart Parking Monitoring & Spatial Analytics
Interactive web interface displaying live occupancy metrics, video feeds,
slot-by-slot dwell times, lane obstruction alerts, bad parking detection,
and separate Near-Miss vs Actual Collision scenarios.
"""

import base64
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import winsound

from src.spatial_engine import SpatialEngine
from src.tracker import VehicleTracker
from src.visualizer import ParkingVisualizer
from src.motion_compensator import CameraMotionCompensator


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
        "config": "configs/commercial_parking_slots.json",
    },
    {
        "id": "r2", "badge": "4K", "kicker": "Real footage",
        "source": "🚁 Vidéo Réelle 2 : Drone Zénithal 90° (Top-Down)",
        "title": "Drone's Eye View",
        "genres": ["Real footage", "Drone", "141 slots"],
        "desc": "A 90° top-down drone pass over a 141-slot lot. Occupancy for every "
                "single slot, seen from above.",
        "video": "4196207-uhd_3840_2160_24fps.mp4",
        "config": "configs/drone_parking_lot.json",
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


@st.cache_data(show_spinner=False)
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
        cap = cv2.VideoCapture(ep["video"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, 90)
        ok, frame = cap.read()
        cap.release()
        return frame if ok else None
    return None


@st.cache_data(show_spinner=False)
def episode_image(ep_id, width):
    frame = episode_frame(ep_id)
    if frame is None:
        return None
    h, w = frame.shape[:2]
    frame = cv2.resize(frame, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


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


model_map = {"yolov8n.pt (Baseline COCO)": "yolov8n.pt"}
if os.path.exists("models"):
    for pt_file in sorted(Path("models").glob("*.pt")):
        fname = pt_file.name
        if "v1" in fname:
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
        video = st.empty()
        alerts = st.empty()
    with col_side:
        st.markdown('<div class="pw-side-title">Slot status</div>', unsafe_allow_html=True)
        table = st.empty()
    return metrics, video, table, alerts


episode = EP_BY_ID[st.session_state.selected]
# A single slot for the page body, so the browse screen is fully replaced while the player loop runs.
with st.empty().container():
    if st.session_state.playing:
        (
            (metric_free, metric_occ, metric_rate, metric_entries, metric_incidents),
            video_placeholder, table_placeholder, alert_placeholder,
        ) = render_player(episode)
    else:
        render_browse(episode)

source_type = episode["source"]
video_path = episode.get("video")
config_file = episode.get("config", SIM_CONFIG)
if episode["id"] == "upload":
    video_path = st.session_state.get("upload_path")
    config_file = st.session_state.get("upload_config", UPLOAD_CONFIGS[0])


def run_dashboard():
    if not os.path.exists(config_file):
        st.error(f"Fichier de configuration '{config_file}' introuvable.")
        return

    engine = SpatialEngine(config_path=config_file)
    visualizer = ParkingVisualizer(slot_definitions=engine.slots)

    tracker = None
    cap = None

    if "Scénario" in source_type:
        pass  # Mode simulation cinématique directe
    elif "Webcam" in source_type:
        cap = cv2.VideoCapture(0)
        tracker = VehicleTracker(model_name=model_choice, conf_threshold=conf_thresh)
    else:
        if not video_path or not os.path.exists(video_path):
            st.warning("Vidéo introuvable ou non sélectionnée.")
            return
        cap = cv2.VideoCapture(video_path)
        tracker = VehicleTracker(model_name=model_choice, conf_threshold=conf_thresh)

    frame_idx = 0
    width, height = 1280, 720

    # Véhicules pour les différents scénarios
    car_101 = {"id": 101, "x": 50.0, "y": 360.0}
    car_102 = {"id": 102, "x": 350.0, "y": 190.0}
    car_104 = {"id": 104, "x": 60.0, "y": 360.0}
    car_105 = {"id": 105, "x": 50.0, "y": 360.0}
    collision_occurred = False
    collision_sound_played = False

    gmc = CameraMotionCompensator(max_features=500)
    base_slots = [dict(s) for s in engine.slots]

    while True:
        curr_time = time.time()

        if "Scénario" in source_type:
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

        else:
            ret, frame = cap.read()
            if not ret:
                st.info("Flux vidéo terminé.")
                break

            if frame.shape[1] > 1280:
                frame = cv2.resize(frame, (1280, 720))

            gmc.update(frame)
            zone_info = gmc.get_active_zone_info()
            comp_slots = gmc.transform_polygons(base_slots, frame_width=frame.shape[1], frame_height=frame.shape[0])
            engine.set_dynamic_slots(comp_slots)
            visualizer.set_dynamic_slots(comp_slots)

            tracked = tracker.track(frame)
            analytics = engine.update(tracked, current_time=curr_time, camera_dx=gmc.camera_dx, camera_dy=gmc.camera_dy)
            vis = visualizer.render(frame, analytics, tracked, zone_info=zone_info, fps=30.0)

        # KPIs
        total = analytics["total_slots"]
        free = analytics["free_count"]
        occ = analytics["occupied_count"]
        rate = analytics["occupancy_rate"]
        entries = analytics.get("total_entries", 0)
        alerts = analytics.get("obstruction_alerts", [])
        straddles = analytics.get("line_straddle_alerts", [])
        collisions = analytics.get("collision_alerts", [])
        total_incidents = len(alerts) + len(straddles) + len(collisions)

        metric_free.metric("Places Libres", f"{free} / {total}")
        metric_occ.metric("Places Occupées", f"{occ}")
        metric_rate.metric("Taux d'Occupation", f"{rate:.1f} %")
        metric_entries.metric("Véhicules Entrés", f"{entries}")
        metric_incidents.metric("Incidents Détectés", f"{total_incidents}")

        # Rendu image Streamlit
        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        video_placeholder.image(vis_rgb, channels="RGB", use_container_width=True)

        # Table des places
        dwell = analytics.get("dwell_data", {})
        states = analytics.get("states", {})
        table_rows = []
        for slot in engine.slots:
            sid = slot["id"]
            is_occ = states.get(sid, False)
            d_info = dwell.get(sid, {})
            elapsed_sec = int(d_info.get("elapsed_seconds", 0))
            mins, secs = divmod(elapsed_sec, 60)
            car_id = d_info.get("car_id", "-")

            status_str = "🔴 Occupée" if is_occ else "🟢 Libre"
            dwell_str = f"{mins:02d}m {secs:02d}s" if is_occ else "-"

            table_rows.append({
                "Place": slot["name"],
                "État": status_str,
                "Véhicule": f"Car #{car_id}" if car_id else "-",
                "Temps Stationné": dwell_str
            })

        table_df = pd.DataFrame(table_rows)
        table_placeholder.dataframe(table_df, use_container_width=True, hide_index=True)

        # Bannières d'alertes avec tous les incidents visibles
        alert_messages = []

        # COLLISION (Highest Priority)
        for col in collisions:
            if col["type"] == "COLLISION":
                alert_messages.append(("error", f"🚨💥 **COLLISION CONFIRMÉE** entre **Car #{col['car1_id']}** x **Car #{col['car2_id']}** !"))
            else:
                # NEAR-MISS (Very Visible)
                alert_messages.append(("warning", f"⚠️⚡ **RISQUE D'ACCROCHAGE DÉTECTÉ** : **Car #{col['car1_id']}** ↔ **Car #{col['car2_id']}** — Rapprochement d'urgence détecté !"))

        # STRADDLE (High Priority)
        for stra in straddles:
            slots_str = ", ".join([f"Slot {sid:02d}" for sid in stra["slot_ids"]])
            alert_messages.append(("error", f"⛔ **MAUVAIS STATIONNEMENT** : **Car #{stra['car_id']}** à cheval sur {slots_str} !"))

        # LANE OBSTRUCTION (Medium Priority)
        for alt in alerts:
            alert_messages.append(("warning", f"⚠️ **VOIE BLOQUÉE** : **Car #{alt['car_id']}** immobilisé ({alt['duration']:.1f}s)"))

        # Display all alerts (errors first, then warnings)
        if alert_messages:
            errors = [msg for typ, msg in alert_messages if typ == "error"]
            warnings = [msg for typ, msg in alert_messages if typ == "warning"]

            if errors:
                alert_placeholder.error(" | ".join(errors))
            elif warnings:
                alert_placeholder.warning(" | ".join(warnings[:2]))  # Show max 2 warnings
        else:
            alert_placeholder.empty()

        frame_idx += 1
        time.sleep(0.03)

    if cap:
        cap.release()


if st.session_state.playing:
    run_dashboard()
