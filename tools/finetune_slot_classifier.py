"""
Fine-tune the PKLot-trained slot classifier on a small real-camera dataset.

PKLot gets the model to a strong starting point (99.1% top-1 on a held-out lot), but
your camera's angle, height, lighting and car mix are a different domain. Rather than
training from scratch on a handful of real labels, continue training from the existing
checkpoint at a low learning rate for a few epochs -- this adapts the model to your
camera without needing thousands of real examples.

Usage:
  python tools/build_slot_dataset.py --source video --video <url_or_file> \
      --config configs/my_cam.json --labels my_cam_labels.csv --group my_cam \
      --out datasets/slots_mycam --val-groups my_cam_val   # a second labeling session, different day

  python tools/finetune_slot_classifier.py --data datasets/slots_mycam \
      --base runs/classify/train/weights/best.pt --out-name finetune_mycam

If datasets/slots_mycam has no val/ split (e.g. you only labeled one session), pass
--val-frac to carve a validation slice out of train instead of training blind.
"""

import argparse
import shutil
from pathlib import Path


def ensure_val_split(data_root: Path, val_frac: float):
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    if val_dir.exists() and any(val_dir.rglob("*.jpg")):
        return
    if val_frac <= 0:
        raise SystemExit(
            f"{val_dir} has no images and --val-frac is 0. Either label a second "
            "session as a separate group for val, or pass --val-frac to carve one out."
        )
    print(f"[INFO] No val split found; moving {val_frac:.0%} of train/ into val/")
    for label_dir in train_dir.iterdir():
        if not label_dir.is_dir():
            continue
        images = sorted(label_dir.glob("*.jpg"))
        n_val = max(1, int(len(images) * val_frac))
        out_dir = val_dir / label_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        for img in images[:n_val]:
            shutil.move(str(img), str(out_dir / img.name))
        print(f"  {label_dir.name}: moved {n_val}/{len(images)} to val")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path, help="Dataset root from build_slot_dataset.py")
    ap.add_argument(
        "--base", default="runs/classify/train/weights/best.pt",
        help="Checkpoint to continue from. Use 'scratch' instead when your camera angle is very "
             "different from PKLot's overhead view (e.g. a handheld eye-level shot of a row of "
             "spots): continuing from PKLot's roof-view features can anchor the model to the "
             "wrong cues (negative transfer) rather than help. 'scratch' starts from ImageNet "
             "weights (yolo11n-cls.pt) instead and needs more of your own labeled images "
             "(several hundred per class, not a few dozen) plus more epochs to converge.",
    )
    ap.add_argument("--epochs", type=int, default=None,
                     help="Default: 15 for fine-tuning from a PKLot checkpoint, 60 for --base scratch")
    ap.add_argument("--lr0", type=float, default=None,
                     help="Default: 0.0005 fine-tuning (avoid forgetting PKLot features), 0.01 for --base scratch")
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default=0)
    ap.add_argument("--val-frac", type=float, default=0.0,
                     help="If --data has no val/ split, carve this fraction out of train/ for it")
    ap.add_argument("--out-name", default="finetune_mycam")
    args = ap.parse_args()

    ensure_val_split(args.data, args.val_frac)

    from ultralytics import YOLO

    from_scratch = args.base == "scratch"
    base_weights = "yolo11n-cls.pt" if from_scratch else args.base
    epochs = args.epochs if args.epochs is not None else (60 if from_scratch else 15)
    lr0 = args.lr0 if args.lr0 is not None else (0.01 if from_scratch else 0.0005)
    if from_scratch:
        print("[INFO] --base scratch: starting from ImageNet weights, not the PKLot checkpoint. "
              "Make sure datasets/<...>/train has at least ~300-500 images per class -- a handful "
              "of real photos will overfit a from-scratch classifier badly.")

    model = YOLO(base_weights)
    results = model.train(
        data=str(args.data),
        epochs=epochs,
        lr0=lr0,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project="runs/classify",
        name=args.out_name,
        exist_ok=True,
        patience=5,  # stop early if val accuracy plateaus -- a handful of real images overfits fast
    )
    print(f"\n[DONE] Fine-tuned weights: runs/classify/{args.out_name}/weights/best.pt")
    print("Copy/point the app at that path (or overwrite runs/classify/train/weights/best.pt) to use it.")


if __name__ == "__main__":
    main()
