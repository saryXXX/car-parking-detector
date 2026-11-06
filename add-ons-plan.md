# Smart Parking: AI add-ons plan

How to add machine learning, deep learning and an LLM to the parts of the project that are still hand-tuned rules: **parking-slot occupancy** and **collision / near-miss detection**.

**Everything in this plan is free:** open datasets, open-source libraries, and open-weight models that run locally on the project laptop (RTX 4060, 8 GB VRAM, 24 GB RAM). No paid API is used anywhere.

Each add-on has:
- why it exists and what it replaces,
- step-by-step instructions,
- a ready-to-paste **prompt for an AI coding assistant** to implement it,
- acceptance criteria you can check before moving on.

The add-ons are ordered by dependency: do them in order.

---

## 0. Where the project stands today

| Part | Current implementation | Weakness |
|---|---|---|
| Vehicle detection | YOLOv8n (COCO) through `src/tracker.py` | Fine on the commercial lot; misses cars in the top-down drone view |
| Drone fine-tune | `models/yolov8n_parking_v1.pt`, classes `car`, `empty_slot`, `occupied_slot` | On a mid-video drone frame it returned 242 `occupied_slot` boxes that overlap into an unusable tangle, and 0 `car` boxes |
| Slot occupancy | Ground point `P_sol` inside the hand-drawn polygon, then 30-frame hysteresis (`src/spatial_engine.py`, thresholds 0.8 / 0.2) | A car hidden by a pedestrian, or a car seen at a steep angle, puts `P_sol` in the wrong place |
| Collision / near-miss | Fixed pixel rules in `src/spatial_engine.py` lines 293–304: overlap > 0.20 or centre distance < 75 px, while closing faster than 50 px/s; near-miss at < 120 px | Pixels are not metres: 75 px near the camera is about 1 m, far from it several metres. The thresholds were tuned on the simulator only |
| Lane obstruction | Immobile > 2.5 s outside any slot | Rule-based; fine as it is |

**Principle for everything below:** keep the current rules as the fallback and as the baseline the jury compares against. Every learned component must beat the rule on the same test data, or it does not ship.

---

## 1. Target architecture

```
video frame
   │
   ├─► YOLO detector ──► ByteTrack ──► tracks (id, bbox, class)
   │                                      │
   │                    [A] homography ───┤  pixel → metres on the ground
   │                                      │
   │    [B] slot crops ──► slot CNN ──────┤  p(occupied) per slot
   │                                      ▼
   │                               spatial_engine.update()
   │                                 ├─ occupancy  (rule OR CNN + hysteresis)
   │                                 ├─ collision  (rule OR [F] ML model)
   │                                 └─ events ──► [D] event log (SQLite)
   │                                                  │
   └─► keyframes of an alert ─────────────────────────┤
                                                      ▼
                                         [G] local LLM via Ollama (async, off the frame loop)
                                           ├─ G1 verify alert (multimodal)
                                           ├─ G2 French incident report
                                           └─ G3 questions on the history
```

New files (all under `src/` unless noted):

| File | Add-on | Purpose |
|---|---|---|
| `homography.py` | A | Image ↔ ground-plane transform per camera |
| `configs/<camera>_homography.json` | A | 4+ reference point pairs per camera |
| `slot_classifier.py` | B | Crop each slot polygon, run the CNN, return p(occupied) |
| `tools/build_slot_dataset.py` | B | Turn PKLot / CNRPark / own videos into a train/val folder tree |
| `event_log.py` | D | Write tracks and events to SQLite |
| `tools/simulate_trajectories.py` | E | Generate labelled synthetic trajectories |
| `trajectory_features.py` | F | Pairwise features (TTC, closing speed, ...) |
| `collision_model.py` | F | Load the trained model; predict collision / near-miss / normal |
| `tools/train_collision_model.py` | F | Training and evaluation script |
| `llm_reviewer.py` | G | Alert verification, incident reports, history questions |

A switch in the Streamlit settings menu chooses **Rules** or **ML** for occupancy and for collisions independently, so the jury can see both on the same video.

---

## A. Ground-plane homography (prerequisite, half a day)

**Why:** every distance, speed and time-to-collision must be in metres and seconds, not pixels. Speeds in pixels per second are also wrong whenever the camera moves.

**Steps**

1. For each fixed camera, pick at least 4 points on the ground whose real positions you know. Parking-line corners are ideal: standard slots are about 2.5 m × 5 m. Measure them on site or use the lot's plan.
2. Store them in `configs/<camera>_homography.json`:
   ```json
   {
     "image_points":  [[412, 655], [1180, 640], [1505, 1010], [160, 1035]],
     "ground_points_m": [[0, 0], [10, 0], [10, 5], [0, 5]]
   }
   ```
3. Compute `H = cv2.findHomography(image_points, ground_points_m, cv2.RANSAC)` once at start-up, and map points with `cv2.perspectiveTransform`.
4. Apply it to `P_sol` (the bottom-centre of each box), never to the box centre: only the ground contact point lies on the plane.
5. For the moving-camera video, transform `P_sol` with the GMC offset first (`gmc.camera_dx/dy`), then apply `H`.
6. Extend the existing calibrator (`src/calibrate.py`) with a mode to click the 4 image points and type their ground coordinates.

**Acceptance:** a parked car's ground position stays within ±0.3 m across 30 frames, and the length of a known slot measures 5 m ± 0.25 m.

**Prompt for the coding assistant**

```text
In this repo (Python, OpenCV, Streamlit), add a ground-plane homography module.

Context:
- src/spatial_engine.py computes the ground contact point of each vehicle with
  get_bottom_contact_point(bbox) -> (x, (y2)) in pixels, and uses pixel thresholds for
  collision (75 px), near-miss (120 px) and closing speed (-50 px/s).
- src/motion_compensator.py exposes camera_dx / camera_dy for the moving-camera video.
- src/calibrate.py is the existing click-to-draw slot calibrator.

Tasks:
1. Create src/homography.py with class GroundPlane:
   - from_json(path) loading {"image_points": [[x,y]x4+], "ground_points_m": [[X,Y]x4+]}
   - image_to_ground(points: np.ndarray[N,2]) -> np.ndarray[N,2] in metres (cv2.perspectiveTransform)
   - ground_to_image(...) for drawing
   - reprojection_error() in metres, printed at load time
   Use cv2.findHomography with RANSAC. Raise a clear error if fewer than 4 points.
2. Add a "--homography" mode to src/calibrate.py: the user clicks 4+ ground points on the
   first frame, types their metric coordinates in the terminal, and the tool writes
   configs/<name>_homography.json.
3. In SpatialEngine.__init__, accept an optional homography_path. When present, keep a
   per-track history of ground positions in metres and expose a helper
   track_kinematics(track_id) -> (position_m, velocity_mps, heading_rad), smoothed over the
   last 5 frames. Do not change any existing behaviour when homography_path is None.
4. Add tests in tests/test_homography.py: an identity-like case, a known rectangle mapping,
   and the "fewer than 4 points" error.
Keep the existing pixel rules untouched; this task only adds metric measurements.
```

---

## B. Per-slot occupancy classifier (deep learning, 2–3 days)

**Why:** instead of asking "is the car's ground point inside the polygon?", ask a CNN "does this slot image contain a car?". It is robust to partial occlusion and steep angles, and public datasets already contain hundreds of thousands of labelled slot crops. Amato et al. (2017) showed a small CNN doing this well enough to run on the cameras themselves.

**Data**

| Source | What it gives | Notes |
|---|---|---|
| PKLot (Almeida et al., 2015) | 12,416 full frames from 3 lots (PUCPR, UFPR04, UFPR05), 695,899 labelled slot crops; sunny, cloudy and rainy days | Daylight only. The license differs by host (CC BY 4.0 on Roboflow; "research use" in the original paper): check the copy you download |
| CNRPark-EXT (Amato et al., 2017) | Slot crops from 9 cameras on a campus, across seasons, with occlusions and difficult angles | Closer to real CCTV conditions than PKLot |
| Your own videos | Crops of your calibrated polygons | Label by hand or semi-automatically (step 3) |

**Steps**

1. **Build the dataset** with `tools/build_slot_dataset.py` into the Ultralytics classification layout:
   ```
   datasets/slots/
     train/occupied/*.jpg   train/empty/*.jpg
     val/occupied/*.jpg     val/empty/*.jpg
     test/occupied/*.jpg    test/empty/*.jpg
   ```
   Split **by parking lot or camera, never by random image.** Neighbouring frames are near-duplicates, so a random split leaks test images into training and inflates accuracy. Keep one PKLot lot and all of your own test video out of training.
2. **Crop your own slots:** for each polygon in `configs/*.json`, warp it to a 128×256 rectangle (`cv2.getPerspectiveTransform` on the 4 corners). This rectifies the perspective, so every crop looks like a top-down slot.
3. **Pre-label your crops** with the current rule (state after hysteresis) and correct them by hand. 1,000–2,000 corrected crops are enough for fine-tuning.
4. **Train in two stages:**
   ```bash
   # stage 1: public data
   yolo classify train data=datasets/slots model=yolo11n-cls.pt imgsz=224 epochs=30 batch=128
   # stage 2: fine-tune on your crops (lower learning rate)
   yolo classify train data=datasets/slots_own model=runs/classify/train/weights/best.pt imgsz=224 epochs=15 lr0=0.001
   ```
   Use the newest `-cls` checkpoint your installed Ultralytics provides (the current docs use `yolo26n-cls.pt`; `yolo11n-cls.pt` also works). An alternative is a torchvision MobileNetV3-Small with a 2-class head.
5. **Integrate:** in `SpatialEngine.update`, replace `Inside_t ∈ {0,1}` with `p_t = p(occupied)` from the CNN, and keep the same hysteresis: `R_k = mean(p_t)` over 30 frames, OCCUPIED at ≥ 0.8, FREE at ≤ 0.2. Run the CNN on all slots as one batch every 3rd frame, reusing the last value in between, to stay real-time.
6. **Fusion option:** OCCUPIED if `0.6·p_cnn + 0.4·inside_rule ≥ threshold`. Report the CNN alone, the rule alone and the fusion.

**Acceptance (on the held-out lot and your test video):** slot-level accuracy ≥ 97% and fewer state flips per slot per minute than the rule. If it does not beat the rule, it stays off by default.

**Prompt for the coding assistant**

```text
Add a per-slot occupancy classifier to this Smart Parking repo.

Context:
- Slots are polygons in configs/*.json: [{"id", "name", "polygon": [[x,y]x4], ...}].
- src/spatial_engine.py SpatialEngine.update() builds, per slot, a deque of 0/1 "inside"
  values (ground point in polygon), then applies hysteresis with occupied_threshold=0.8 and
  free_threshold=0.2 over hysteresis_window=30.
- Ultralytics is installed (YOLO classify task available).

Tasks:
1. tools/build_slot_dataset.py
   - Input A: a PKLot or CNRPark-EXT folder (slot crops already labelled by folder name or
     XML; support PKLot's "Occupied"/"Empty" folder convention).
   - Input B: a video + slot config + a CSV of manual labels (frame, slot_id, occupied).
   - For B, rectify each 4-point polygon to 128x256 with cv2.getPerspectiveTransform.
   - Output the Ultralytics classify layout (train|val|test)/(occupied|empty)/*.jpg.
   - The split is by a --group key (lot name or camera), never random per image. Print class
     counts per split.
2. src/slot_classifier.py with class SlotClassifier:
   - __init__(weights_path, device="cpu", every_n_frames=3)
   - predict(frame, slots) -> dict[slot_id, float]: crop and rectify every slot, run ONE
     batched inference, return p(occupied). Cache results between runs of every_n_frames.
3. In SpatialEngine, add an optional slot_classifier argument and a mode
   occupancy_mode in {"rule", "cnn", "fusion"} (default "rule"). In "cnn", push p(occupied)
   into the per-slot deque instead of 0/1; in "fusion", push 0.6*p + 0.4*inside. The hysteresis
   code stays identical. Dwell timers and straddle detection must keep working in all modes.
4. tools/eval_occupancy.py: given a video, a slot config and a ground-truth CSV
   (frame, slot_id, occupied), report for each mode: accuracy, F1, and state flips per slot
   per minute. Print a markdown table.
5. Add a Streamlit selectbox "Occupation : Règles / CNN / Fusion" in the existing
   "Model settings" popover of app.py and pass it to SpatialEngine.
Do not train anything in this task; only write the code and a short README section with the
two training commands.
```

---

## C. Oriented detector for the drone view (deep learning, 2 days)

**Why:** in a top-down view, slots and cars are diagonal rectangles packed edge to edge. Axis-aligned boxes around diagonal objects overlap massively, which is exactly the tangle the V1 model produced. Oriented bounding boxes (OBB) fit rotated objects tightly. Ultralytics OBB models are pretrained on DOTAv1, an aerial dataset that includes vehicle classes, so they start from the right kind of imagery.

**Steps**

1. Extract about 300 frames from the drone video at different times (`extract_frames.py` already exists); keep the last 20% of the video for testing only.
2. Label rotated boxes for `car` (and optionally `empty_slot`), in the YOLO-OBB format: `class x1 y1 x2 y2 x3 y3 x4 y4`, normalised. Pre-label with `semi_auto_annotate.py` adapted to output 4 corners, then correct by hand (CVAT and Label Studio both support rotated boxes).
3. Train:
   ```bash
   yolo obb train data=configs/drone_obb.yaml model=yolo11n-obb.pt imgsz=1024 epochs=100
   ```
   (or the newest `-obb` checkpoint available). Use `imgsz` ≥ 1024: drone cars are small.
4. Prediction returns `xywhr` boxes; the ground point of a top-down car is simply its centre.
5. Compare on the held-out frames against `yolov8n_parking_v1.pt`: mAP50, the count error per frame (detected vs true cars), and visual inspection.

**Acceptance:** count error ≤ 5% per frame on held-out frames, and boxes that do not overlap each other.

**Prompt for the coding assistant**

```text
Add oriented-bounding-box (OBB) support for the top-down drone video.

Context: models/yolov8n_parking_v1.pt (axis-aligned, classes car/empty_slot/occupied_slot)
produces heavily overlapping boxes on 4196207-uhd_3840_2160_24fps.mp4. src/tracker.py wraps
an Ultralytics YOLO model with ByteTrack and returns dicts {"id","bbox","conf","cls_name"}.

Tasks:
1. tools/make_obb_dataset.py: sample N frames (default 300) from a video, write them to
   datasets/drone_obb/images/{train,val,test} with a time-based split (last 20% = test), and
   create configs/drone_obb.yaml for the Ultralytics OBB task.
2. Update semi_auto_annotate.py with an --obb flag that pre-labels with an existing OBB model
   and writes YOLO-OBB label files (class + 4 normalised corners).
3. In src/tracker.py, detect when the loaded model's task is "obb". Then convert each
   rotated box to: "bbox" = its axis-aligned bounds (for compatibility), plus a new
   "corners" key (4x2 array) and "angle". The ground point for OBB detections is the box
   centre. ByteTrack still runs on the axis-aligned bounds.
4. In src/visualizer.py, draw "corners" as a rotated polygon when present.
5. tools/eval_drone_counts.py: on the test frames with labels, print mAP50 and mean absolute
   count error per frame for any list of model paths.
```

---

## D. Track and event logging (data collection, half a day)

**Why:** the collision model (E/F) and the LLM layer (G) both need history. The app currently forgets everything after each frame.

**Schema (SQLite, `data/events.db`)**

```sql
CREATE TABLE tracks (
  run_id TEXT, frame INTEGER, t REAL, track_id INTEGER, cls TEXT, conf REAL,
  x1 REAL, y1 REAL, x2 REAL, y2 REAL,           -- pixels
  gx REAL, gy REAL,                              -- ground metres (NULL without homography)
  slot_id INTEGER                                -- NULL when not in a slot
);
CREATE TABLE events (
  event_id INTEGER PRIMARY KEY, run_id TEXT, t_start REAL, t_end REAL,
  type TEXT CHECK(type IN ('ENTRY','EXIT','PARKED','LEFT','DWELL_EXCEEDED',
                           'STRADDLE','OBSTRUCTION','NEAR_MISS','COLLISION')),
  track_ids TEXT, slot_ids TEXT, details_json TEXT,
  source TEXT,                                   -- 'rule' | 'ml'
  llm_verdict TEXT, llm_confidence REAL, llm_report TEXT
);
CREATE TABLE runs (run_id TEXT PRIMARY KEY, video TEXT, config TEXT, model TEXT, started_at TEXT);
```

Write in batches (every 30 frames) so logging never slows the frame loop. For each COLLISION or NEAR_MISS, also save 5 keyframes (from −1.0 s to +0.5 s around the event) as JPEGs under `data/events/<event_id>/` for add-on G.

**Prompt for the coding assistant**

```text
Add persistent logging to the Smart Parking app.
1. src/event_log.py with class EventLog(db_path="data/events.db"): creates the three tables
   below if missing, start_run(video, config, model) -> run_id, log_tracks(frame, t, tracks)
   buffered and flushed every 30 frames or on close(), log_event(...) -> event_id, and
   save_keyframes(event_id, frames: list[np.ndarray]) writing JPEGs to data/events/<id>/.
   [paste the SQL schema from add-ons-plan.md section D]
2. Keep a ring buffer of the last 45 frames in app.py's run_dashboard loop. When
   analytics["collision_alerts"] contains a new alert (deduplicate by the pair of track ids
   within 3 s), log the event and save 5 keyframes spanning -1.0 s to +0.5 s around it (wait
   for the +0.5 s frames before saving).
3. Log ENTRY/EXIT (tripwire), PARKED/LEFT (slot state changes), DWELL_EXCEEDED, STRADDLE and
   OBSTRUCTION from the existing analytics dict. Store details as JSON.
4. Logging must never crash the video loop: catch and print sqlite errors.
```

---

## E. Synthetic trajectory generator (training data for collisions, 1 day)

**Why:** you will not film real crashes, and real accident datasets are dash-cam or highway footage (DoTA, CADP), far from a slow car park. A simulator gives unlimited labelled examples of exactly your situation: reversing out of a slot, a car passing in the aisle, braking or not. It extends the five scenarios in `app.py` with randomness.

**Generator design**

- Work directly in **metres** on a 2D lot plan (aisle 6 m wide, slots 2.5 m × 5 m); cars are oriented rectangles of about 1.8 m × 4.5 m.
- Per sample, randomise:
  - the reversing car's start slot, speed (0.5–2.5 m/s) and reaction (brakes or not, and when),
  - the aisle car's start position, speed (2–6 m/s) and whether it brakes,
  - noise: ±0.15 m on positions (detector jitter), random missing frames (occlusion), 10–30 FPS.
- **Label each sample from the geometry**, not from a rule:
  - `COLLISION` if the two rectangles intersect;
  - `NEAR_MISS` if the minimum gap < 1.0 m, or the minimum TTC < 1.5 s, without contact;
  - `NORMAL` otherwise.
- Record the label **and the time of first contact or closest approach**, so lead time can be measured.
- Generate about 20,000 pair-sequences of 4–8 s with balanced classes; hold out whole parameter ranges for testing (for example, all aisle speeds > 5 m/s) to test generalisation, not memorisation.
- Add real `NORMAL` pairs from your videos via the event log (D) to keep false alarms realistic.

**Prompt for the coding assistant**

```text
Create tools/simulate_trajectories.py, a synthetic data generator for car-park conflicts.

World: metric 2D plane. One aisle along x (y in [0, 6] m), slots 2.5 m wide x 5 m deep on
both sides. Cars are oriented rectangles 1.8 m x 4.5 m with a simple kinematic bicycle model
(max accel 2.5 m/s^2, max braking 6 m/s^2).

Each sample has two cars:
- Car A reverses out of a random slot into the aisle with speed U(0.5, 2.5) m/s. With
  probability p_brake (default 0.5) it starts braking at a random moment when the other car
  is within U(3, 12) m.
- Car B drives along the aisle with speed U(2, 6) m/s from a random x offset, and brakes
  with probability 0.3 when the TTC to A drops below U(1, 3) s.
Simulate at a random FPS in {10, 15, 25, 30} for 4-8 s. Add Gaussian position noise
(sigma 0.15 m), randomly drop 0-10% of frames, and apply a constant-velocity Kalman filter
to the noisy positions to mimic the tracker.

Labels from exact geometry (use shapely polygon intersection on the noise-free states):
COLLISION if the rectangles ever intersect; else NEAR_MISS if min distance < 1.0 m or
min TTC < 1.5 s; else NORMAL. Also store t_event (first contact or closest approach).

Output: data/sim/trajectories.parquet with columns
sample_id, frame, t, car (A|B), x, y, vx, vy, heading, label, t_event, split.
The split is by parameter range (CLI flag --holdout "B_speed>5") plus a random 10%
validation set. Add --n (default 20000) and --seed. Print class balance per split.
Also export 20 random samples as small matplotlib GIFs to data/sim/preview/ for checking.
```

---

## F. Learned collision and near-miss model (ML, 2–3 days)

### F1. Features (per pair of tracks, per frame, over the last 1 s)

All values are in metres and seconds (add-on A):

| Feature | Definition |
|---|---|
| `d` | Distance between the two cars (polygon gap if OBB, else centre distance minus 2 m) |
| `v_close` | Closing speed = −d(d)/dt |
| `ttc` | Time to collision under constant velocity (below); capped at 10 s |
| `ttc_min_1s` | Minimum TTC over the last second |
| `heading_rel` | Angle between the two velocity vectors |
| `speed_a`, `speed_b` | Speeds |
| `decel_a`, `decel_b` | Maximum braking over the last 0.5 s (a sudden stop is a strong signal) |
| `overlap_iou` | Image-box IoU (catches what the ground projection misses) |
| `in_aisle_a`, `in_aisle_b` | Whether each car is in the aisle polygon |
| `reversing_a`, `reversing_b` | Moving against its heading (reversing out of a slot) |

**Time to collision** (cars approximated as discs of radius r ≈ 1.2 m): with relative position **p** and relative velocity **v**, TTC is the smallest t > 0 solving |**p** + **v**·t| = 2r:

```
a = v·v,   b = 2 p·v,   c = p·p − (2r)²
disc = b² − 4ac
TTC = (−b − √disc) / (2a)   if disc ≥ 0 and the result is > 0, else ∞
```

TTC and post-encroachment time (PET) are the standard *surrogate safety measures* used by traffic engineers to count conflicts without waiting for crashes (FHWA's SSAM tool). Citing them shows the jury that the near-miss definition comes from the traffic-safety field, not from guesswork.

### F2. Models, from simplest to strongest

1. **Baseline:** the current pixel rules, plus the same rules converted to metres (a fair second baseline).
2. **LightGBM classifier** on the feature table, with 3 classes (NORMAL, NEAR_MISS, COLLISION), `class_weight="balanced"`. Fast, works with little data, and its feature importances explain decisions to the jury ("TTC and braking matter most").
3. **GRU sequence model** (optional): input is the last 30 frames of the pair's features; 2 GRU layers × 64 units, then a 3-class softmax. Use it if LightGBM plateaus.
4. **Anomaly detector** (optional, needs no crash data): an LSTM autoencoder trained only on real NORMAL pairs from your videos; a high reconstruction error flags an unusual interaction. Use it as a second signal, not alone.
5. **Trajectory forecasting** (stretch goal): predict each car's next 1.5 s with a small LSTM, in the spirit of Social LSTM (Alahi et al., 2016), and compute TTC on the predicted paths, so the warning comes earlier.

### F3. Evaluation

On the held-out simulated range **and** on real clips (your test video plus the simulated scenarios played through the real pipeline):

| Metric | Why it matters |
|---|---|
| Precision and recall per class | Standard |
| False alarms per hour on real NORMAL footage | What an operator actually suffers |
| Lead time: seconds between the first alert and contact | An early warning is the point of near-miss detection |
| Confusion matrix | Shows NEAR_MISS ↔ COLLISION confusions |

Decision rule: alert when `p(COLLISION) ≥ 0.6`, or `p(NEAR_MISS) + p(COLLISION) ≥ 0.5` for 3 consecutive frames (a small hysteresis, as for slots). Tune both thresholds on the validation set only.

**Prompt for the coding assistant**

```text
Implement a learned collision / near-miss detector.

Inputs available:
- data/sim/trajectories.parquet from tools/simulate_trajectories.py (columns: sample_id,
  frame, t, car, x, y, vx, vy, heading, label, t_event, split).
- At runtime, SpatialEngine exposes track_kinematics(track_id) -> position_m, velocity_mps,
  heading (src/homography.py, add-on A).

Tasks:
1. src/trajectory_features.py: function pair_features(hist_a, hist_b, fps) -> dict computing,
   over the last 1 s: d, v_close, ttc (disc model, r=1.2 m, capped at 10 s), ttc_min_1s,
   heading_rel, speed_a/b, decel_a/b (max over the last 0.5 s), reversing_a/b, in_aisle_a/b
   (aisle polygon passed in). Pure numpy, unit-tested in tests/test_features.py: head-on
   approach, parallel motion (ttc = inf), stationary pair.
2. tools/train_collision_model.py:
   - Build a per-frame feature table from the parquet; label each frame with the sample label
     only from 2 s before t_event onward (earlier frames = NORMAL), so the model learns to
     anticipate.
   - Train LightGBM (multiclass, class_weight balanced, early stopping on the val split).
   - Evaluate on the holdout split: per-class precision/recall, confusion matrix, mean lead
     time, and the same metrics for the existing rules (port the pixel rules to metres:
     75 px -> 1.0 m, 120 px -> 1.6 m, -50 px/s -> -0.7 m/s). Write reports/collision_eval.md.
   - Save models/collision_lgbm.txt and a JSON with feature names and thresholds.
3. src/collision_model.py: class CollisionModel(model_path) with
   predict(pairs: list[(track_a, track_b, features)]) -> list[{"type", "p_collision",
   "p_near_miss"}], applying 3-frame hysteresis per pair.
4. In SpatialEngine, add collision_mode in {"rule", "ml"} (default "rule"). In "ml", build
   features for every pair closer than 8 m and use CollisionModel; the alert dicts must keep
   the same keys as today ("type", "car1_id", "car2_id", ...) so app.py does not change.
5. Add the "Collisions : Règles / ML" selectbox to the Model settings popover in app.py.
Keep each module small and dependency-light (numpy, pandas, lightgbm, shapely).
```

---

## G. LLM layer with a free local model (event level, 2 days)

**Rule:** the LLM never runs per frame. It runs on **events** (seconds of latency are fine) and on **questions**. Detection stays with the vision models.

### Model and runtime

- **Runtime: [Ollama](https://ollama.com)**, free and open source. It runs open-weight models on the laptop's GPU and exposes a local HTTP API (`http://localhost:11434`). Nothing leaves the machine, so no account, key or cost is involved.
- **Model: `qwen3.5:4b`** (Qwen3.5, open weights, about 3.3–4 GB). It is the newest Qwen family, natively multimodal (reads images and text), writes good French, and handles all three tasks below, so only one model has to be downloaded. At this size it leaves room on the RTX 4060's 8 GB for YOLO running at the same time. Qwen3.5 "thinks" before answering by default; pass `think=False` for these short structured tasks to keep them fast.
- **Bigger option:** `qwen3.5:9b` (about 6.6–7.6 GB) is stronger but nearly fills the 8 GB, so only use it when YOLO is not running (for example, for reports written after a run).
- **Alternatives to compare** (also free on Ollama): `qwen3-vl:4b` / `qwen3-vl:8b` (the previous Qwen vision models, 3.3 / 6.2 GB; may not show in the Ollama app's model picker but `ollama pull` works), `gemma3:4b` (vision, multilingual).
- **Setup:** keep Ollama up to date (Qwen3.5 needs a recent release).
  ```bash
  # 1. install Ollama for Windows from https://ollama.com/download, then:
  ollama pull qwen3.5:4b
  ollama run qwen3.5:4b "Bonjour, réponds en une phrase."   # smoke test
  # 2. Python client
  pip install ollama pydantic
  ```
- **Structured answers:** pass a Pydantic schema as `format=Model.model_json_schema()` and set `options={"temperature": 0}`; Ollama constrains the output to that JSON schema, and you validate it with `Model.model_validate_json(response.message.content)`.
- **Images:** pass file paths (or bytes) in the message's `images` list.
- **Speed:** expect a few seconds per alert check on the 4060, which is fine because it runs in a background thread. Unload the model when a run ends (`keep_alive=0` on the last call) to give the GPU back to YOLO.
- **Honest limits:** a small local model is weaker than the large commercial models. That is why G1 is evaluated on hand-labelled alerts before it is trusted, and why it can only tag alerts, never hide them.
- **Privacy:** because the model runs locally, keyframes never leave the laptop, which is a strong argument for the defence. Still blur faces and plates in the keyframes you **store** (`data/events/`). Personal data processing in Tunisia falls under the organic law n° 2004-63 on personal data protection.

### G1. Alert verification (multimodal second opinion)

Flow: a COLLISION or NEAR_MISS event fires → 5 blurred keyframes + the event JSON go to the model → its verdict is stored in `events.llm_verdict` and shown next to the alert in the dashboard. The vision alert is shown immediately; the verdict arrives a few seconds later.

**System prompt (runtime, paste as-is)**

```text
You review safety alerts from a car-park video monitoring system. A detector and a
tracker have flagged a possible collision or near-miss between two vehicles. You receive
5 chronological keyframes (about 1 s before to 0.5 s after the flagged moment) and a JSON
summary of the tracker's measurements.

Decide what actually happened between the two vehicles named in the JSON (their boxes are
drawn on the frames with their ids):
- "collision": the vehicles make physical contact.
- "near_miss": no contact, but one vehicle had to brake or swerve, or they came within
  roughly one metre while moving toward each other.
- "false_alarm": nothing dangerous. Typical causes: perspective making distant cars overlap
  in the image, a shadow or reflection, a tracker id switch, a parked car next to a slowly
  moving one, an occlusion by a pedestrian or a pillar.
- "uncertain": the frames do not show enough to decide.

Base the decision on what the frames show. Use the JSON as supporting evidence: its
distances and speeds come from an automatic system and can be wrong, especially when a
vehicle is partly hidden. Do not identify people or read number plates.
Write the explanation in French, two sentences at most, naming the visual evidence.
```

**User turn:** the 5 images in order, then:

```text
Événement à vérifier :
{"event_id": 418, "type_detecte": "COLLISION", "vehicules": [101, 105],
 "distance_min_m": 0.4, "vitesse_rapprochement_mps": 1.9, "ttc_min_s": 0.3,
 "freinage_max_mps2": {"101": 0.2, "105": 4.8}, "source": "ml", "p_collision": 0.81}
```

**Code**

```python
from pathlib import Path
from typing import Literal

from ollama import Client, ResponseError
from pydantic import BaseModel, Field, ValidationError

MODEL = "qwen3.5:4b"
SYSTEM_PROMPT = Path("prompts/alert_review_system.txt").read_text(encoding="utf-8")
client = Client(host="http://localhost:11434")


class AlertVerdict(BaseModel):
    verdict: Literal["collision", "near_miss", "false_alarm", "uncertain"]
    confidence: float = Field(ge=0, le=1)
    false_alarm_cause: Literal["none", "perspective", "shadow_reflection", "id_switch",
                               "occlusion", "parked_vehicle", "other"]
    explication: str


def review_alert(keyframe_paths: list[Path], event_json: str) -> AlertVerdict | None:
    try:
        response = client.chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",
                 "content": f"Événement à vérifier :\n{event_json}",
                 "images": [str(p) for p in keyframe_paths]},
            ],
            format=AlertVerdict.model_json_schema(),
            think=False,
            options={"temperature": 0},
        )
        return AlertVerdict.model_validate_json(response.message.content)
    except (ResponseError, ValidationError, ConnectionError) as exc:
        print(f"LLM review unavailable: {exc}")
        return None
```

Small local models handle five images less reliably than one. If verdicts are erratic, tile the 5 keyframes into one image strip (`np.hstack` after resizing each to 384 px high) with the frame time written on each tile, and send that single image.

**Evaluate G1 before relying on it:** collect 100+ logged alerts, label them by hand, and measure how many false alarms the LLM removes and how many true collisions it wrongly dismisses. A dismissed true collision is the costly error: if it happens, use the LLM verdict only to add a "probable false alarm" tag, never to hide an alert.

### G2. French incident report

Triggered when an event closes, or on demand from the dashboard. Input: the event row and the related track summary, as JSON. Output: a short report for the parking operator.

**System prompt (runtime)**

```text
Tu rédiges des rapports d'incident pour l'exploitant d'un parking surveillé par caméra.
Tu reçois un événement détecté automatiquement (JSON) et, s'il existe, l'avis d'un
vérificateur visuel.

Règles :
- Français professionnel, phrases courtes, vouvoiement absent (rapport factuel).
- N'invente aucun fait absent du JSON. Si une information manque, écris « non déterminé ».
- Les mesures (distances, vitesses) sont des estimations automatiques : présente-les comme
  telles (« environ », « estimé à »).
- Ne mentionne ni personnes identifiables ni plaques d'immatriculation.

Format exact :
### Incident n° {event_id} — {type en clair}
**Heure :** hh:mm:ss (durée)
**Lieu :** allée ou place concernée
**Véhicules :** identifiants de suivi
**Déroulé :** 2 à 4 phrases chronologiques.
**Mesures estimées :** liste courte.
**Vérification visuelle :** verdict et confiance, ou « non effectuée ».
**Action recommandée :** une phrase (ex. vérifier les images, dégager la voie, aucune).
```

Call it with `client.chat(model="qwen3.5:4b", think=False, messages=[{"role": "system", "content": REPORT_PROMPT}, {"role": "user", "content": event_json}], options={"temperature": 0.2})` and store `response.message.content` in `events.llm_report`. Check every report against its JSON during evaluation: small models occasionally invent details, which is why the prompt forbids it and asks for « non déterminé ». Add a "Rapport" button per alert in the dashboard, and an export of all reports of a run as one Markdown file.

### G3. Questions on the history (text-to-SQL, read-only)

Examples: « Combien de dépassements de durée hier après 14 h ? », « Quelle place est la plus occupée cette semaine ? », « Liste les quasi-collisions de la vidéo 2 ».

Design for safety:
1. The model writes **one SQL `SELECT`** for the schema of section D (given in the system prompt). Ask for it as JSON with `format={"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"]}` so there is no extra text to strip.
2. Your code validates it: a single statement starting with `SELECT`, no `;` inside, only the three known tables. It runs on a connection opened read-only (`sqlite3.connect("file:data/events.db?mode=ro", uri=True)`), with a row limit and a 2 s timeout.
3. The rows go back to the model, which answers in French and shows the query it used.

**System prompt (runtime)**

```text
You answer questions about a car-park monitoring history stored in SQLite.
Schema:
[paste the CREATE TABLE statements from section D]
Times in tables are seconds from the start of a run; runs.started_at is the wall-clock
start (ISO 8601). Event types: ENTRY, EXIT, PARKED, LEFT, DWELL_EXCEEDED, STRADDLE,
OBSTRUCTION, NEAR_MISS, COLLISION.

Step 1: reply with exactly one SQLite SELECT statement that answers the question, and
nothing else. Use only the tables above. Add LIMIT 200 unless the query aggregates.
Step 2: you will then receive the rows. Answer in French in one to three sentences, then
show the SQL you used in a code block. If the rows cannot answer the question, say so.
```

**Prompt for the coding assistant (G1–G3)**

```text
Add an LLM layer to the Smart Parking app using a FREE LOCAL model through Ollama.
Do not use any paid or cloud API. Use the official `ollama` Python package
(pip install ollama), host http://localhost:11434, model name read from the env var
PARKING_LLM_MODEL with default "qwen3.5:4b" (pass think=False on every call).

1. Save the three system prompts from add-ons-plan.md section G as
   prompts/alert_review_system.txt, prompts/incident_report_system.txt and
   prompts/history_qa_system.txt.
2. src/llm_reviewer.py:
   - is_available() -> bool: True when the Ollama server answers and the model is pulled
     (client.list()); cache the result for 60 s.
   - review_alert(event_id): load data/events/<id>/*.jpg (5 keyframes, already blurred when
     saved), call client.chat with role "system" + a user message carrying the event JSON and
     the image paths in "images", format=AlertVerdict.model_json_schema(),
     options={"temperature": 0}; validate with AlertVerdict.model_validate_json and store
     verdict/confidence/explanation in the events table. Add a --tile option that merges the
     5 frames into one horizontal strip before sending.
   - write_report(event_id): build the event JSON (plus the verdict if present), call
     client.chat with the report system prompt, temperature 0.2, store events.llm_report.
   - ask_history(question): two-step text-to-SQL. Step 1 asks for {"sql": "..."} via the
     format parameter. Validate that the SQL is a single SELECT on the known tables, run it on
     a read-only sqlite URI connection with a 200-row limit and a 2 s timeout, then send the
     rows back for the French answer. Return (answer, sql, rows).
   - Catch ollama.ResponseError, pydantic.ValidationError and connection errors separately;
     never let an LLM failure stop the video loop. On the last call of a run pass
     keep_alive=0 to free the GPU.
3. Run review_alert in a background thread (concurrent.futures.ThreadPoolExecutor, 1 worker,
   because the GPU is shared with YOLO) so the dashboard never waits on the model.
4. In app.py: next to each alert, show the LLM verdict badge when available ("Confirmé",
   "Fausse alerte probable", "Incertain"); add a "Rapport" button per alert that shows
   write_report's output; add a popover "Questions" text box calling ask_history.
   When is_available() is False, hide the LLM features and show one line:
   "Assistant IA local indisponible : lancez Ollama puis `ollama pull qwen3.5:4b`."
5. tools/eval_llm_review.py: given a CSV of hand labels (event_id, true_label), run
   review_alert on each and print the confusion matrix and the number of true collisions
   dismissed.
```

---

## H. Dashboard integration (1 day)

- **Model settings popover:** Occupation (Règles / CNN / Fusion), Collisions (Règles / ML), Vérification LLM (on/off).
- **Alerts panel:** each alert shows its source (rule or ML), its probability, and the LLM verdict once it arrives.
- **"Comparer" mode** for the defence: run rules and ML side by side on the same video, with two counters (alerts raised, alerts confirmed).
- **Reports tab:** the run's incident reports and an export button.

---

## I. Evaluation protocol for the defence

1. **Never train or tune on the unseen test video** ("Vidéo 2 · test inédit" in the deck). Touch it once, at the end.
2. Report one table per task, with the same test data for every row:

| Occupancy | Accuracy | F1 | Flips / slot / min |
|---|---|---|---|
| Rule (ground point + hysteresis) | | | |
| CNN + hysteresis | | | |
| Fusion | | | |

| Collisions | Precision | Recall | False alarms / h | Mean lead time (s) |
|---|---|---|---|---|
| Pixel rules (current) | | | | |
| Metric rules | | | | |
| LightGBM | | | | |
| LightGBM + LLM check | | | | |

3. Show 2–3 failure cases honestly (the jury will ask). The drone V1 tangle is a good "before" picture for add-on C.

---

## J. Schedule

| Project phase | Add-ons | Estimated effort |
|---|---|---|
| Phase 3 · Détection et tracking | A homography, C drone OBB, D logging | 3–4 days |
| Phase 4 · Logique métier et UI | B slot CNN, E simulator, F LightGBM, H dashboard | 6–8 days |
| Phase 4–5 | G LLM layer (G1 first, then G2, G3 if time allows) | 2–3 days |
| Phase 5 · Test inédit et soutenance | Section I evaluation on the unseen video; optional GRU / forecasting | 2 days |

Between two people, this matches the deck's split: student 1 takes A, C, E, F (vision and tracking); student 2 takes B, D, G, H (business logic and interface).

---

## K. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Simulated collisions do not transfer to real video | Hold out parameter ranges; validate on real NORMAL footage for false alarms; keep the rules as the fallback |
| No real collision footage to test recall | Stage near-misses safely with two cars at walking speed and a spotter (clear it with your supervisor), or replay simulated trajectories through the real pipeline |
| Homography wrong on a sloped lot | Check the reprojection error at load time; use 6+ points spread over the whole view |
| Slot CNN fails at night | PKLot is daylight-only; include CNRPark-EXT and your own evening clips, or state the limitation |
| LLM dismisses a real collision | The LLM only tags alerts and never hides them; measure dismissed true collisions (G1 evaluation) |
| Local model too slow, or GPU shared with YOLO | One background worker; `qwen3.5:4b` by default (`qwen3.5:2b` if still too heavy); unload with `keep_alive=0`; LLM features are optional and hidden when Ollama is not running |
| Small model invents details in reports | Temperature 0–0.2, « non déterminé » rule in the prompt, spot-check reports against their JSON |
| Privacy | The model runs locally, so frames never leave the laptop; blur faces and plates in stored keyframes; mention law n° 2004-63 in the report |

---

## L. Dependencies

```bash
pip install lightgbm pyarrow shapely ollama pydantic
# local LLM runtime (free): install Ollama from https://ollama.com/download, then
ollama pull qwen3.5:4b
# optional sequence models
pip install torch torchvision
```

Ultralytics (already installed) covers classification (`-cls`) and oriented detection (`-obb`).

---

## M. Sources

Datasets
- Almeida, P., Oliveira, L. S., Britto Jr., A. S., Silva Jr., E. J., Koerich, A. L. (2015). *PKLot – A robust dataset for parking lot classification.* Expert Systems with Applications 42(11), 4937–4949. DOI [10.1016/j.eswa.2015.02.009](https://doi.org/10.1016/j.eswa.2015.02.009). Paper PDF: https://www.inf.ufpr.br/lesoliveira/download/ESWA2015.pdf · Copy on Roboflow: https://public.roboflow.com/object-detection/pklot
- Amato, G., Carrara, F., Falchi, F., Gennaro, C., Meghini, C., Vairo, C. (2017). *Deep learning for decentralized parking lot occupancy detection.* Expert Systems with Applications 72, 327–334. DOI [10.1016/j.eswa.2016.10.055](https://doi.org/10.1016/j.eswa.2016.10.055). Draft: https://falchi.isti.cnr.it/Draft/2017-ESWA-Draft.pdf · Code: https://github.com/fabiocarrara/deep-parking
- Yao, Y. et al. (2023). *DoTA: Unsupervised Detection of Traffic Anomaly in Driving Videos.* IEEE TPAMI 45(1), 444–459. Preprint: https://arxiv.org/abs/2004.03044 · Dataset: https://github.com/MoonBlvd/Detection-of-Traffic-Anomaly (dash-cam footage; useful for ideas, not a direct match for a car park)
- Shah, A., Lamare, J.-B., Nguyen Anh, T., Hauptmann, A. (2018). *CADP: A Novel Dataset for CCTV Traffic Camera based Accident Analysis.* IEEE AVSS 2018. https://arxiv.org/abs/1809.05782 (fixed CCTV road accidents; closer in viewpoint, different setting)

Detection and tracking
- Ultralytics, *Oriented Bounding Box (OBB) task*: https://docs.ultralytics.com/tasks/obb · OBB dataset format: https://docs.ultralytics.com/datasets/obb
- Ultralytics, *Classification dataset layout*: https://docs.ultralytics.com/datasets/classify
- Zhang, Y. et al. (2022). *ByteTrack: Multi-Object Tracking by Associating Every Detection Box.* ECCV 2022. https://arxiv.org/abs/2110.06864

Geometry and traffic safety
- OpenCV, *Basic concepts of the homography explained with code*: https://docs.opencv.org/4.13.0/d9/dab/tutorial_homography.html
- Gettman, D., Pu, L., Sayed, T., Shelby, S. (2008). *Surrogate Safety Assessment Model and Validation: Final Report.* FHWA-HRT-08-051. https://www.fhwa.dot.gov/publications/research/safety/08051/ (TTC and PET as conflict measures)
- FHWA (2017). *Surrogate Safety Assessment Model (SSAM) Version 3.0* TechBrief, FHWA-HRT-17-027. https://www.fhwa.dot.gov/publications/research/operations/17027/17027.pdf

Machine learning
- Ke, G. et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS 30. https://proceedings.neurips.cc/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf
- Alahi, A. et al. (2016). *Social LSTM: Human Trajectory Prediction in Crowded Spaces.* CVPR 2016, 961–971. https://openaccess.thecvf.com/content_cvpr_2016/html/Alahi_Social_LSTM_Human_CVPR_2016_paper.html

Local LLM (free, open weights)
- Ollama, download and docs: https://ollama.com/download · https://docs.ollama.com
- Ollama structured outputs (JSON schema via `format`, vision example): https://docs.ollama.com/capabilities/structured-outputs · announcement: https://ollama.com/blog/structured-outputs
- Qwen3.5 on Ollama (0.8B–122B, text + image, thinking toggle; 4B ≈ 3.3–4 GB, 9B ≈ 6.6–7.6 GB): https://ollama.com/library/qwen3.5
- Qwen3-VL on Ollama (alternative; 4B ≈ 3.3 GB, 8B ≈ 6.2 GB): https://ollama.com/library/qwen3-vl
- Ollama Python library: https://github.com/ollama/ollama-python

Background already cited in the Phase 1 deck
- Shoup, D. (2006). *Cruising for parking.* Transport Policy 13(6), 479–486 (the ~30% of downtown traffic figure).
