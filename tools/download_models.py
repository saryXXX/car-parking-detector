"""
Download the aerial vehicle detectors used for drone / top-down footage into models/.

    python tools/download_models.py            # default: geo-trax (recommended)
    python tools/download_models.py --all      # also VisDrone YOLOv8s

Weights are git-ignored; run this once after cloning.
"""
import argparse
import urllib.request
from pathlib import Path

MODELS = {
    # name: (Hugging Face repo, file in repo, license)
    "geotrax_yolov8s_1920.pt": ("rfonod/geo-trax", "geotrax_hbb_yolov8s_1920_v1.pt", "CC-BY-4.0"),
    "visdrone_yolov8s.pt": ("dronefreak/visdrone-yolov8s", "best.pt", "AGPL-3.0"),
}
DEFAULT = ["geotrax_yolov8s_1920.pt"]


def download(name: str, out_dir: Path) -> None:
    repo, fname, lic = MODELS[name]
    dst = out_dir / name
    if dst.exists():
        print(f"[SKIP] {dst} already exists")
        return
    url = f"https://huggingface.co/{repo}/resolve/main/{fname}"
    print(f"[GET ] {url}  ({lic})")
    tmp = dst.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dst)
    print(f"[ OK ] {dst} ({dst.stat().st_size / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="Download every model, not just the default")
    ap.add_argument("--out", default="models")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    for name in (MODELS if args.all else DEFAULT):
        download(name, out)


if __name__ == "__main__":
    main()
