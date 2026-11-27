"""
Process-wide cache of loaded YOLO models + background GPU warm-up.

Loading weights and the first CUDA inference (context creation, kernel loading) cost
~17-20 s on a cold process; every later run would otherwise reload the weights again.
Modules stay imported across Streamlit reruns, so a module-level cache survives them.

Models are cached per (path, role): a model used by `model.track()` registers tracking
callbacks on itself, so it must not be shared with plain `model()` detection calls.
"""

import threading
from typing import Dict, Iterable, Tuple

import numpy as np

_models: Dict[Tuple[str, str], object] = {}
_lock = threading.Lock()
_warm_done = threading.Event()
_warm_started = False
_warm_error: str = ""


def get_yolo(path: str, role: str = "track"):
    """Loaded YOLO for `path`. role: "track" (VehicleTracker), "detect" (aerial tiles),
    "classify" (slot classifier) -- one instance per role.

    Getting a "track" model resets its tracker state, so two VehicleTrackers on the SAME
    weights must not run interleaved in one process (sequential runs are fine)."""
    key = (str(path), role)
    with _lock:
        model = _models.get(key)
        if model is None:
            from ultralytics import YOLO
            model = YOLO(str(path))
            _models[key] = model
    if role == "track":
        # Drop trackers left by a previous run so IDs restart at 1 and no stale tracks carry over;
        # Ultralytics creates fresh ones on the next track(persist=True) call.
        predictor = getattr(model, "predictor", None)
        if predictor is not None and hasattr(predictor, "trackers"):
            del predictor.trackers
    return model


def _warmup(specs: Iterable[Tuple[str, str, int]]):
    global _warm_error
    try:
        for path, role, imgsz in specs:
            model = get_yolo(path, role)
            dummy = np.zeros((720, 1280, 3), np.uint8)
            model.predict(dummy, imgsz=imgsz, verbose=False)
    except Exception as exc:  # warm-up is best effort: a real run will surface the error
        _warm_error = f"{type(exc).__name__}: {exc}"
    finally:
        _warm_done.set()


def start_warmup(specs: Iterable[Tuple[str, str, int]]):
    """Load and run one dummy inference per (path, role, imgsz) in a background thread.
    Safe to call on every rerun: only the first call starts the thread."""
    global _warm_started
    with _lock:
        if _warm_started:
            return
        _warm_started = True
    threading.Thread(target=_warmup, args=(list(specs),), daemon=True, name="yolo-warmup").start()


def warmup_ready() -> bool:
    return _warm_done.is_set() or not _warm_started


def wait_warmup(timeout: float = 120.0) -> bool:
    """Block until the warm-up finishes (so a run doesn't race it for the same model)."""
    return _warm_done.wait(timeout) if _warm_started else True
