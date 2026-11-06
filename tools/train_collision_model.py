"""
Train the LightGBM collision/near-miss model on synthetic trajectories (add-on F2/F3).

Builds a per-frame feature table from data/sim/trajectories.parquet (one row per pair
per frame), labels each frame NORMAL unless it falls within 2 s of the sample's
t_event (in which case it takes the sample's label), trains a 3-class LightGBM model,
evaluates it against the held-out parameter range AND against the existing pixel rule
converted to metres, and writes a markdown report plus the saved model.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `src.*` imports

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src.trajectory_features import FEATURE_NAMES, TrackHistory, TrackSample, pair_features

LABEL_LOOKAHEAD_S = 2.0
LABELS = ["NORMAL", "NEAR_MISS", "COLLISION"]
LABEL_TO_INT = {l: i for i, l in enumerate(LABELS)}

# Pixel-rule-to-metres conversion (from the existing SpatialEngine thresholds), used as
# the second, fairer baseline alongside the raw pixel rule (F2.1)
PX_TO_M = {"collision_dist_m": 1.0, "near_miss_dist_m": 1.6, "closing_speed_mps": 0.7}


def build_feature_table(parquet_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    rows = []
    for sample_id, sdf in df.groupby("sample_id"):
        label = sdf["label"].iloc[0]
        t_event = sdf["t_event"].iloc[0]
        split = sdf["split"].iloc[0]
        fps = len(sdf[sdf.car == "A"]) / max(1e-6, sdf[sdf.car == "A"].t.max() - sdf[sdf.car == "A"].t.min() + 1e-6)

        a_rows = sdf[sdf.car == "A"].sort_values("frame")
        b_rows = sdf[sdf.car == "B"].sort_values("frame")
        common_frames = sorted(set(a_rows.frame) & set(b_rows.frame))
        hist_a, hist_b = TrackHistory(), TrackHistory()
        a_by_frame = a_rows.set_index("frame")
        b_by_frame = b_rows.set_index("frame")

        for f in common_frames:
            ra, rb = a_by_frame.loc[f], b_by_frame.loc[f]
            hist_a.push(TrackSample(t=ra.t, x=ra.x, y=ra.y, heading=ra.heading))
            hist_b.push(TrackSample(t=rb.t, x=rb.x, y=rb.y, heading=rb.heading))
            if len(hist_a.samples) < 2 or len(hist_b.samples) < 2:
                continue
            feats = pair_features(hist_a, hist_b, fps=fps)

            if label != "NORMAL" and t_event is not None and (t_event - ra.t) <= LABEL_LOOKAHEAD_S:
                frame_label = label
            else:
                frame_label = "NORMAL"

            feats.update(sample_id=sample_id, frame=f, split=split, label=frame_label)
            rows.append(feats)
    return pd.DataFrame(rows)


def rule_baseline_predict(df: pd.DataFrame) -> np.ndarray:
    """Pixel rule ported to metres: collision if d<=1.0m closing >0.7 m/s; near-miss if d<=1.6m."""
    pred = np.full(len(df), LABEL_TO_INT["NORMAL"])
    near = (df["d"].values <= PX_TO_M["near_miss_dist_m"])
    coll = (df["d"].values <= PX_TO_M["collision_dist_m"]) & (df["v_close"].values >= PX_TO_M["closing_speed_mps"])
    pred[near] = LABEL_TO_INT["NEAR_MISS"]
    pred[coll] = LABEL_TO_INT["COLLISION"]
    return pred


def mean_lead_time(df: pd.DataFrame, pred: np.ndarray) -> float:
    """Average seconds between the first non-NORMAL prediction and the sample's t_event."""
    df = df.copy()
    df["pred"] = pred
    leads = []
    sim = pd.read_parquet("data/sim/trajectories.parquet")
    t_events = sim.drop_duplicates("sample_id").set_index("sample_id")["t_event"]
    fps_lookup = sim.groupby("sample_id")["frame"].max()
    for sid, g in df[df.pred != LABEL_TO_INT["NORMAL"]].groupby("sample_id"):
        t_event = t_events.get(sid)
        if t_event is None or pd.isna(t_event):
            continue
        first_alert_frame = g["frame"].min()
        frame_t = sim[(sim.sample_id == sid) & (sim.car == "A") & (sim.frame == first_alert_frame)]["t"]
        if len(frame_t):
            leads.append(t_event - frame_t.values[0])
    return float(np.mean(leads)) if leads else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=Path, default=Path("data/sim/trajectories.parquet"))
    ap.add_argument("--out-model", type=Path, default=Path("models/collision_lgbm.txt"))
    ap.add_argument("--out-report", type=Path, default=Path("reports/collision_eval.md"))
    args = ap.parse_args()

    print("Building per-frame feature table (this walks every sample)...")
    feat_df = build_feature_table(args.parquet)
    feat_df["y"] = feat_df["label"].map(LABEL_TO_INT)
    print(f"Feature table: {len(feat_df)} rows")
    print(feat_df.groupby(["split", "label"]).size().unstack(fill_value=0))

    train = feat_df[feat_df.split == "train"]
    val_holdout = feat_df[feat_df.split == "val_holdout"]
    val_random = feat_df[feat_df.split == "val_random"]

    X_train, y_train = train[FEATURE_NAMES], train["y"]
    X_val, y_val = val_random[FEATURE_NAMES], val_random["y"]

    # lgb.train() is the native Booster API, not the sklearn wrapper: it has no
    # class_weight="balanced" option, so per-sample weights are computed by hand
    # (inverse class frequency, like sklearn's "balanced" scheme) and passed to Dataset.
    class_counts = y_train.value_counts()
    n_classes, n_samples = len(class_counts), len(y_train)
    class_weight = {c: n_samples / (n_classes * cnt) for c, cnt in class_counts.items()}
    train_weight = y_train.map(class_weight).values

    train_set = lgb.Dataset(X_train, label=y_train, weight=train_weight)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

    params = dict(
        objective="multiclass", num_class=3,
        learning_rate=0.05, num_leaves=31, verbose=-1,
    )
    model = lgb.train(
        params, train_set, num_boost_round=500, valid_sets=[val_set],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )

    args.out_model.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(args.out_model))
    meta = {"feature_names": FEATURE_NAMES, "labels": LABELS,
            "alert_threshold_collision": 0.6, "alert_threshold_near_or_collision": 0.5,
            "alert_hysteresis_frames": 3}
    with open(args.out_model.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    report_lines = ["# Collision / near-miss model evaluation\n",
                     f"Trained on `{args.parquet}`, {len(train)} train rows, "
                     f"{len(val_holdout)} held-out-range rows, {len(val_random)} random-val rows.\n",
                     "Features used (metric simulator only; `overlap_iou` from the full F1 "
                     "spec needs real pixel boxes and is added at runtime integration, not here).\n"]

    for name, split_df in [("Held-out parameter range (B_speed>5)", val_holdout),
                            ("Random validation split", val_random)]:
        if len(split_df) == 0:
            continue
        X, y = split_df[FEATURE_NAMES], split_df["y"]
        pred_lgbm = model.predict(X).argmax(axis=1)
        pred_rule = rule_baseline_predict(split_df)

        report_lines.append(f"\n## {name}\n")
        for model_name, pred in [("LightGBM", pred_lgbm), ("Rule (ported to metres)", pred_rule)]:
            report_lines.append(f"\n### {model_name}\n")
            report_lines.append("```\n" + classification_report(
                y, pred, labels=[0, 1, 2], target_names=LABELS, zero_division=0) + "\n```\n")
            cm = confusion_matrix(y, pred, labels=[0, 1, 2])
            report_lines.append("Confusion matrix (rows=true, cols=pred, order NORMAL/NEAR_MISS/COLLISION):\n")
            report_lines.append("```\n" + str(cm) + "\n```\n")
            lead = mean_lead_time(split_df, pred)
            report_lines.append(f"Mean lead time before t_event: {lead:.2f} s\n")

    importances = dict(zip(FEATURE_NAMES, model.feature_importance(importance_type="gain").tolist()))
    top = sorted(importances.items(), key=lambda kv: -kv[1])[:6]
    report_lines.append("\n## Feature importance (top 6, gain)\n")
    for name, val in top:
        report_lines.append(f"- `{name}`: {val:.0f}\n")

    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text("".join(report_lines), encoding="utf-8")
    print(f"\nModel -> {args.out_model}")
    print(f"Report -> {args.out_report}")


if __name__ == "__main__":
    main()
