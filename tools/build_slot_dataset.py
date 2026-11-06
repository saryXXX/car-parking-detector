"""
Build a per-slot occupied/empty image classification dataset in the Ultralytics
`yolo classify` layout:

    <out>/train/occupied/*.jpg   <out>/train/empty/*.jpg
    <out>/val/occupied/*.jpg     <out>/val/empty/*.jpg
    <out>/test/occupied/*.jpg    <out>/test/empty/*.jpg

Two input modes:

  --source pklot <PKLot_root>
      PKLot.tar.gz layout: <root>/<lot>/<weather>/<date>/*.jpg plus matching *.xml
      per frame listing each slot polygon and its "occupied"/"empty" label (the XML
      <space id=... occupied="0|1"><rotatedRect>...</rotatedRect></space> format; we
      crop the axis-aligned bbox of the rotated rect, which is what PKLot distributions
      on Roboflow / Kaggle also ship as separate "Occupied"/"Empty" folders -- that
      folder form is also supported, see --source pklot-folders below).

  --source pklot-folders <root>
      Already-cropped PKLot/CNRPark-EXT style folders:
      <root>/<lot>/<weather_or_camera>/{Occupied,Empty}/*.jpg (case-insensitive).

  --source video <video> --config <slots.json> --labels <labels.csv> --group <name>
      labels.csv columns: frame,slot_id,occupied (occupied is 0/1). For each labelled
      (frame, slot_id) the 4-point polygon from the slot config is perspective-warped
      to a fixed 128x256 rectangle.

The split is always by --group (lot name / camera / video name), never by random
image, to avoid leaking near-duplicate neighbouring frames between train and val/test.
Pass --val-groups and --test-groups to pin specific groups to val/test; anything left
over goes to train.
"""

import argparse
import csv
import random
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

CROP_SIZE = (128, 256)  # (width, height) after perspective rectification


def save_crop(img, out_dir: Path, label: str, split: str, stem: str):
    d = out_dir / split / label
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])


def assign_split(group: str, val_groups: set, test_groups: set) -> str:
    if group in test_groups:
        return "test"
    if group in val_groups:
        return "val"
    return "train"


# ---------------------------------------------------------------------------
# Mode 1: raw PKLot XML (frame + per-slot rotated rect + occupied flag)
# ---------------------------------------------------------------------------

def rotated_rect_to_quad(center, size, angle_deg):
    (cx, cy), (w, h) = center, size
    rect = ((cx, cy), (w, h), angle_deg)
    box = cv2.boxPoints(rect)
    return box.astype(np.float32)


def warp_quad(img, quad: np.ndarray) -> np.ndarray:
    dst = np.array(
        [[0, 0], [CROP_SIZE[0] - 1, 0], [CROP_SIZE[0] - 1, CROP_SIZE[1] - 1], [0, CROP_SIZE[1] - 1]],
        dtype=np.float32,
    )
    # Order quad as TL, TR, BR, BL by angle around centroid
    c = quad.mean(axis=0)
    ang = np.arctan2(quad[:, 1] - c[1], quad[:, 0] - c[0])
    quad = quad[np.argsort(ang)]
    M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return cv2.warpPerspective(img, M, CROP_SIZE)


def build_from_pklot_xml(root: Path, out_dir: Path, val_groups: set, test_groups: set, limit_per_class: int):
    counts = Counter()
    xml_files = sorted(root.rglob("*.xml"))
    if not xml_files:
        print(f"[WARN] No .xml files found under {root}; is this really raw PKLot?")
    random.shuffle(xml_files)
    for xml_path in xml_files:
        img_path = xml_path.with_suffix(".jpg")
        if not img_path.exists():
            continue
        # group = top-level lot folder name, e.g. PKLot/<lot>/<weather>/<date>/img.jpg
        rel = xml_path.relative_to(root).parts
        group = rel[0] if rel else "unknown"
        split = assign_split(group, val_groups, test_groups)

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        try:
            tree = ET.parse(xml_path)
        except ET.ParseError:
            continue
        for space in tree.getroot().findall("space"):
            occ = space.get("occupied")
            if occ is None:
                continue
            label = "occupied" if occ == "1" else "empty"
            if limit_per_class and counts[(split, label)] >= limit_per_class:
                continue
            rect = space.find("rotatedRect")
            if rect is None:
                continue
            cx = float(rect.find("center").get("x"))
            cy = float(rect.find("center").get("y"))
            w = float(rect.find("size").get("w"))
            h = float(rect.find("size").get("h"))
            angle = float(rect.find("angle").get("d"))
            quad = rotated_rect_to_quad((cx, cy), (w, h), angle)
            crop = warp_quad(img, quad)
            stem = f"{group}_{xml_path.stem}_{space.get('id')}"
            save_crop(crop, out_dir, label, split, re.sub(r"[^\w\-]", "_", stem))
            counts[(split, label)] += 1
    return counts


# ---------------------------------------------------------------------------
# Mode 2: already-cropped Occupied/Empty folders (PKLot-on-Roboflow, CNRPark-EXT)
# ---------------------------------------------------------------------------

def build_from_cropped_folders(root: Path, out_dir: Path, val_groups: set, test_groups: set, limit_per_class: int):
    counts = Counter()
    for label_dir in root.rglob("*"):
        if not label_dir.is_dir():
            continue
        name = label_dir.name.lower()
        if name in ("occupied", "busy"):
            label = "occupied"
        elif name in ("empty", "free", "vacant"):
            label = "empty"
        else:
            continue
        rel = label_dir.relative_to(root).parts
        group = rel[0] if rel else "unknown"
        split = assign_split(group, val_groups, test_groups)
        images = list(label_dir.glob("*.jpg")) + list(label_dir.glob("*.png"))
        random.shuffle(images)
        for img_path in images:
            if limit_per_class and counts[(split, label)] >= limit_per_class:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            crop = cv2.resize(img, CROP_SIZE, interpolation=cv2.INTER_AREA)
            stem = re.sub(r"[^\w\-]", "_", f"{group}_{img_path.stem}")
            save_crop(crop, out_dir, label, split, stem)
            counts[(split, label)] += 1
    return counts


# ---------------------------------------------------------------------------
# Mode 3: your own video + slot polygons + hand-labelled CSV
# ---------------------------------------------------------------------------

def build_from_video(video: Path, config: Path, labels: Path, group: str, out_dir: Path,
                      val_groups: set, test_groups: set):
    import json

    counts = Counter()
    slots = {s["id"]: s["polygon"] for s in json.load(open(config, encoding="utf-8"))}
    split = assign_split(group, val_groups, test_groups)

    rows_by_frame: dict[int, list[tuple[int, int]]] = {}
    with open(labels, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows_by_frame.setdefault(int(row["frame"]), []).append(
                (int(row["slot_id"]), int(row["occupied"]))
            )

    cap = cv2.VideoCapture(str(video))
    dst = np.array(
        [[0, 0], [CROP_SIZE[0] - 1, 0], [CROP_SIZE[0] - 1, CROP_SIZE[1] - 1], [0, CROP_SIZE[1] - 1]],
        dtype=np.float32,
    )
    for frame_idx in sorted(rows_by_frame):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        for slot_id, occupied in rows_by_frame[frame_idx]:
            poly = slots.get(slot_id)
            if poly is None:
                continue
            src = np.array(poly[:4], dtype=np.float32)
            M = cv2.getPerspectiveTransform(src, dst)
            crop = cv2.warpPerspective(frame, M, CROP_SIZE)
            label = "occupied" if occupied else "empty"
            stem = f"{group}_f{frame_idx}_s{slot_id}"
            save_crop(crop, out_dir, label, split, stem)
            counts[(split, label)] += 1
    cap.release()
    return counts


def print_report(counts: Counter):
    print("\nClass counts per split:")
    splits = sorted({s for s, _ in counts})
    for split in splits:
        occ = counts.get((split, "occupied"), 0)
        emp = counts.get((split, "empty"), 0)
        print(f"  {split:5s}  occupied={occ:6d}  empty={emp:6d}  total={occ + emp:6d}")
    total = sum(counts.values())
    print(f"  TOTAL  {total}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["pklot", "pklot-folders", "video"], required=True)
    ap.add_argument("--root", type=Path, help="Dataset root for pklot / pklot-folders")
    ap.add_argument("--video", type=Path, help="Video file for --source video")
    ap.add_argument("--config", type=Path, help="Slot polygon config (configs/*.json) for --source video")
    ap.add_argument("--labels", type=Path, help="CSV frame,slot_id,occupied for --source video")
    ap.add_argument("--group", type=str, help="Group name for --source video (e.g. camera/lot id)")
    ap.add_argument("--out", type=Path, default=Path("datasets/slots"))
    ap.add_argument("--val-groups", nargs="*", default=[], help="Group names reserved for validation")
    ap.add_argument("--test-groups", nargs="*", default=[], help="Group names reserved for test")
    ap.add_argument("--limit-per-class", type=int, default=0,
                     help="Cap crops per (split,label) to keep training time reasonable; 0 = no cap")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    val_groups, test_groups = set(args.val_groups), set(args.test_groups)

    if args.source == "pklot":
        counts = build_from_pklot_xml(args.root, args.out, val_groups, test_groups, args.limit_per_class)
    elif args.source == "pklot-folders":
        counts = build_from_cropped_folders(args.root, args.out, val_groups, test_groups, args.limit_per_class)
    else:
        counts = build_from_video(args.video, args.config, args.labels, args.group, args.out,
                                   val_groups, test_groups)

    print_report(counts)
    print(f"\nDataset written to: {args.out.resolve()}")


if __name__ == "__main__":
    main()
