"""
Synthetic trajectory generator for car-park conflicts (add-on E).

World: metric 2D plane. One aisle along x (y in [0, 6] m), slots 2.5 m wide x 5 m deep
on both sides. Cars are oriented rectangles 1.8 m x 4.5 m with a simple kinematic model
(bounded acceleration / braking).

Two cars per sample:
  - Car A reverses out of a random slot into the aisle (speed 0.5-2.5 m/s), and may
    brake when car B gets close.
  - Car B drives along the aisle (speed 2-6 m/s), and may brake when TTC to A drops
    below a threshold.

Labels come from exact geometry on the noise-free trajectory (shapely), never from a
hand-tuned rule:
  COLLISION  if the two rectangles ever intersect
  NEAR_MISS  if min distance < 1.0 m or min TTC < 1.5 s, without contact
  NORMAL     otherwise

Output: data/sim/trajectories.parquet, columns:
  sample_id, frame, t, car (A|B), x, y, vx, vy, heading, label, t_event, split

`split` is "train" by default; samples matching --holdout (e.g. "B_speed>5") are
marked "val_holdout" to test generalisation to parameter ranges never trained on,
plus a random 10% of the remaining samples are marked "val_random".
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Polygon

CAR_LEN, CAR_WID = 4.5, 1.8
AISLE_Y = 6.0
SLOT_DEPTH, SLOT_WIDTH = 5.0, 2.5
MAX_ACCEL, MAX_BRAKE = 2.5, 6.0
NEAR_MISS_DIST_M = 1.0
NEAR_MISS_TTC_S = 1.5


def car_polygon(x, y, heading):
    """Axis-free rectangle centred at (x, y), rotated by heading (radians)."""
    hl, hw = CAR_LEN / 2, CAR_WID / 2
    corners = np.array([[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]])
    c, s = np.cos(heading), np.sin(heading)
    R = np.array([[c, -s], [s, c]])
    rotated = corners @ R.T
    return Polygon(rotated + np.array([x, y]))


def ttc_discs(p_rel, v_rel, r=1.2):
    """Time to collision for two discs of radius r, relative position/velocity p, v."""
    a = v_rel @ v_rel
    b = 2 * p_rel @ v_rel
    c = p_rel @ p_rel - (2 * r) ** 2
    if a < 1e-9:
        return np.inf
    disc = b * b - 4 * a * c
    if disc < 0:
        return np.inf
    t = (-b - np.sqrt(disc)) / (2 * a)
    return t if t > 0 else np.inf


def simulate_one(rng: np.random.Generator, sample_id: int, p_brake_a=0.5, p_brake_b=0.3):
    fps = int(rng.choice([10, 15, 25, 30]))
    duration = rng.uniform(4, 8)
    n_steps = int(duration * fps)
    dt = 1.0 / fps

    # Car A: reverses out of a slot (starts in a slot, moves toward the aisle centre line)
    slot_x = rng.uniform(-10, 10)
    a_speed = rng.uniform(0.5, 2.5)
    a_heading = np.pi / 2  # moving toward +y (out of slot) initially
    ax, ay = slot_x, SLOT_DEPTH
    a_braking_from = None
    a_brake_trigger_dist = rng.uniform(3, 12) if rng.random() < p_brake_a else None

    # Car B: drives along the aisle, at a randomised lane offset (not always dead-centre,
    # so many pairs pass by at a safe lateral gap instead of always being on a collision course)
    b_speed0 = rng.uniform(2, 6)
    b_x0 = slot_x + rng.uniform(-25, 25)
    bx, by = b_x0, rng.uniform(1.8, AISLE_Y - 1.8)
    b_heading = 0.0 if rng.random() < 0.5 else np.pi
    b_dir = 1.0 if b_heading == 0.0 else -1.0
    b_speed = b_speed0
    b_ttc_trigger = rng.uniform(1, 3) if rng.random() < p_brake_b else None

    rows = []
    min_dist, min_ttc, event_frame, collided = np.inf, np.inf, None, False
    av, bv = a_speed, b_speed

    for f in range(n_steps):
        t = f * dt
        p_rel = np.array([bx - ax, by - ay])
        v_a = np.array([0.0, av])
        v_b = np.array([b_dir * bv, 0.0])
        v_rel = v_b - v_a
        dist = float(np.linalg.norm(p_rel)) - 2 * 1.2  # approx surface gap
        ttc = ttc_discs(p_rel, v_rel)
        min_dist = min(min_dist, max(0.0, dist))
        min_ttc = min(min_ttc, ttc)

        poly_a = car_polygon(ax, ay, a_heading)
        poly_b = car_polygon(bx, by, b_heading)
        if poly_a.intersects(poly_b):
            collided = True
            event_frame = event_frame or f
            break
        if dist < NEAR_MISS_DIST_M or ttc < NEAR_MISS_TTC_S:
            event_frame = event_frame if event_frame is not None else f

        rows.append((sample_id, f, t, "A", ax, ay, 0.0, av, a_heading))
        rows.append((sample_id, f, t, "B", bx, by, b_dir * bv, 0.0, b_heading))

        # Car A braking logic
        if a_brake_trigger_dist is not None and dist < a_brake_trigger_dist and a_braking_from is None:
            a_braking_from = f
        if a_braking_from is not None:
            av = max(0.0, av - MAX_BRAKE * dt)
        else:
            av = min(a_speed, av + MAX_ACCEL * dt)
        ay += av * dt

        # Car B braking logic
        if b_ttc_trigger is not None and ttc < b_ttc_trigger:
            bv = max(0.0, bv - MAX_BRAKE * dt)
        else:
            bv = min(b_speed0, bv + MAX_ACCEL * dt)
        bx += b_dir * bv * dt

    if collided:
        label = "COLLISION"
    elif min_dist < NEAR_MISS_DIST_M or min_ttc < NEAR_MISS_TTC_S:
        label = "NEAR_MISS"
    else:
        label = "NORMAL"
    t_event = (event_frame / fps) if event_frame is not None else None

    df = pd.DataFrame(rows, columns=["sample_id", "frame", "t", "car", "x", "y", "vx", "vy", "heading"])
    df["label"] = label
    df["t_event"] = t_event

    # Sensor noise + dropped frames, applied after geometry-based labelling
    noise = rng.normal(0, 0.15, size=(len(df), 2))
    df["x"] += noise[:, 0]
    df["y"] += noise[:, 1]
    drop_frac = rng.uniform(0, 0.10)
    keep = rng.random(len(df)) > drop_frac
    df = df[keep].reset_index(drop=True)

    meta = dict(A_speed=a_speed, B_speed=b_speed0, fps=fps)
    return df, meta


def parse_holdout(expr: str):
    m = re.match(r"\s*(\w+)\s*(>=|<=|>|<|==)\s*([\d.]+)\s*", expr)
    if not m:
        raise ValueError(f"Bad --holdout expression: {expr}")
    field, op, val = m.group(1), m.group(2), float(m.group(3))
    ops = {">": lambda s: s > val, "<": lambda s: s < val, ">=": lambda s: s >= val,
           "<=": lambda s: s <= val, "==": lambda s: s == val}
    return field, ops[op]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--holdout", type=str, default="B_speed>5",
                     help="Parameter-range holdout, e.g. 'B_speed>5' (field: A_speed, B_speed, fps)")
    ap.add_argument("--out", type=Path, default=Path("data/sim/trajectories.parquet"))
    ap.add_argument("--preview", type=int, default=20, help="Number of sample preview GIFs (0 to skip)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    field, holdout_fn = parse_holdout(args.holdout)

    all_rows, metas = [], []
    for sid in range(args.n):
        df, meta = simulate_one(rng, sid)
        all_rows.append(df)
        metas.append(meta)

    full = pd.concat(all_rows, ignore_index=True)
    meta_df = pd.DataFrame(metas)
    meta_df["sample_id"] = range(args.n)

    is_holdout = holdout_fn(meta_df[field])
    rand_val = rng.random(args.n) < 0.10
    split_by_sample = np.where(is_holdout, "val_holdout", np.where(rand_val, "val_random", "train"))
    split_map = dict(zip(meta_df["sample_id"], split_by_sample))
    full["split"] = full["sample_id"].map(split_map)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(args.out, index=False)

    print(f"Wrote {len(full)} rows, {args.n} samples -> {args.out}")
    label_per_sample = full.drop_duplicates("sample_id")[["sample_id", "label", "split"]]
    print("\nClass balance per split:")
    print(label_per_sample.groupby(["split", "label"]).size().unstack(fill_value=0))

    if args.preview:
        preview_dir = args.out.parent / "preview"
        preview_dir.mkdir(exist_ok=True)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.animation as animation
            import matplotlib.pyplot as plt

            sample_ids = rng.choice(full["sample_id"].unique(), size=min(args.preview, args.n), replace=False)
            for sid in sample_ids:
                sdf = full[full.sample_id == sid]
                label = sdf["label"].iloc[0]
                fig, ax = plt.subplots(figsize=(6, 2))
                ax.set_xlim(sdf.x.min() - 5, sdf.x.max() + 5)
                ax.set_ylim(-1, AISLE_Y + 1)
                ax.set_title(f"sample {sid}: {label}")
                dots = {c: ax.plot([], [], "o", ms=10, label=c)[0] for c in ("A", "B")}
                frames = sorted(sdf.frame.unique())

                def update(f, sdf=sdf, dots=dots):
                    for c, dot in dots.items():
                        row = sdf[(sdf.car == c) & (sdf.frame == f)]
                        if len(row):
                            dot.set_data([row.x.values[0]], [row.y.values[0]])
                    return list(dots.values())

                anim = animation.FuncAnimation(fig, update, frames=frames, interval=1000 / 15)
                anim.save(preview_dir / f"sample_{sid}.gif", writer="pillow", fps=15)
                plt.close(fig)
            print(f"Wrote {len(sample_ids)} preview GIFs -> {preview_dir}")
        except ImportError:
            print("[skip] matplotlib not available, skipping preview GIFs")


if __name__ == "__main__":
    main()
