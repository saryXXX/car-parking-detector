"""
Replay cache: a video is analysed once, then replayed at its native frame rate.

The pipeline runs at ~3-4 fps on the 4K demo clips, far below their 24/30 fps. While a
clip is analysed, every annotated frame (JPEG, exactly what was shown) and the dashboard
numbers for that frame are recorded. The next Play of the same clip with the same settings
streams the recording instead of re-running detection.

The key covers everything that changes the output: video file, slot config, model weights,
detector settings, occupancy / collision modes, and the source code of src/ -- editing the
pipeline invalidates old recordings automatically.

Layout of data/cache/replays/<key>/:
    frames.bin   annotated JPEG frames, back to back
    index.json   {"meta": {...}, "offsets": [...], "records": [...]}
"""

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional

CACHE_DIR = Path("data/cache/replays")
KEEP_REPLAYS = 12  # oldest recordings beyond this are deleted


def _file_sig(path: Optional[str]) -> Optional[list]:
    if not path or not os.path.exists(path):
        return None
    st = os.stat(path)
    return [os.path.abspath(path), st.st_size, int(st.st_mtime)]


def _code_fingerprint() -> str:
    h = hashlib.sha1()
    for p in sorted(Path(__file__).parent.glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def make_key(video: str, config: str, model: str, extra_files: List[Optional[str]] = (), **settings) -> str:
    payload = {
        "video": _file_sig(video), "config": _file_sig(config), "model": _file_sig(model) or model,
        "extra": [_file_sig(p) for p in extra_files], "settings": settings, "code": _code_fingerprint(),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:20]


def ui_record(analytics: Dict) -> Dict:
    """The per-frame numbers the dashboard shows, in a compact JSON-safe form."""
    occupied = [int(sid) for sid, on in analytics.get("states", {}).items() if on]
    dwell = analytics.get("dwell_data", {})
    return {
        "total": int(analytics["total_slots"]),
        "free": int(analytics["free_count"]),
        "occ": int(analytics["occupied_count"]),
        "rate": float(analytics["occupancy_rate"]),
        "entries": int(analytics.get("total_entries", 0)),
        "occupied": occupied,
        "dwell": {str(sid): [None if dwell.get(sid, {}).get("car_id") is None else int(dwell[sid]["car_id"]),
                             float(dwell.get(sid, {}).get("elapsed_seconds", 0.0))] for sid in occupied},
        "collisions": [{"type": c["type"], "car1_id": int(c["car1_id"]), "car2_id": int(c["car2_id"])}
                       for c in analytics.get("collision_alerts", [])],
        "straddles": [{"car_id": int(s["car_id"]), "slot_ids": [int(i) for i in s["slot_ids"]]}
                      for s in analytics.get("line_straddle_alerts", [])],
        "obstructions": [{"car_id": int(o["car_id"]), "duration": float(o["duration"])}
                         for o in analytics.get("obstruction_alerts", [])],
    }


class ReplayWriter:
    """Records frames into a temporary folder; `commit()` publishes it atomically, anything
    else (Back button, error, end of session) leaves no partial recording behind."""

    def __init__(self, key: str, meta: Dict):
        self.final = CACHE_DIR / key
        self.tmp = CACHE_DIR / f"{key}.part{os.getpid()}"
        shutil.rmtree(self.tmp, ignore_errors=True)
        self.tmp.mkdir(parents=True)
        self.meta = meta
        self._bin = open(self.tmp / "frames.bin", "wb")
        self.offsets: List[int] = []
        self.records: List[Dict] = []
        self.done = False

    def add(self, jpeg: bytes, record: Dict):
        self.offsets.append(self._bin.tell())
        self._bin.write(jpeg)
        self.records.append(record)

    def commit(self):
        self.offsets.append(self._bin.tell())  # end of the last frame
        self._bin.close()
        (self.tmp / "index.json").write_text(
            json.dumps({"meta": self.meta, "offsets": self.offsets, "records": self.records}), encoding="utf-8")
        shutil.rmtree(self.final, ignore_errors=True)
        os.replace(self.tmp, self.final)
        self.done = True
        prune()

    def abort(self):
        if not self.done:
            self._bin.close()
            shutil.rmtree(self.tmp, ignore_errors=True)


class Replay:
    def __init__(self, folder: Path):
        index = json.loads((folder / "index.json").read_text(encoding="utf-8"))
        self.meta: Dict = index["meta"]
        self.offsets: List[int] = index["offsets"]
        self.records: List[Dict] = index["records"]
        self.fps: float = self.meta["fps"]
        self.frame_count = len(self.records)
        self._bin = open(folder / "frames.bin", "rb")

    def frame(self, i: int) -> bytes:
        self._bin.seek(self.offsets[i])
        return self._bin.read(self.offsets[i + 1] - self.offsets[i])

    def close(self):
        self._bin.close()


def load(key: str) -> Optional[Replay]:
    folder = CACHE_DIR / key
    if not (folder / "index.json").exists():
        return None
    try:
        replay = Replay(folder)
    except (OSError, ValueError, KeyError):
        shutil.rmtree(folder, ignore_errors=True)  # corrupt: analyse again
        return None
    os.utime(folder)  # recently used: survives pruning
    return replay


def prune(keep: int = KEEP_REPLAYS):
    if not CACHE_DIR.exists():
        return
    done = [p for p in CACHE_DIR.iterdir() if p.is_dir() and (p / "index.json").exists()]
    for old in sorted(done, key=lambda p: p.stat().st_mtime, reverse=True)[keep:]:
        shutil.rmtree(old, ignore_errors=True)
