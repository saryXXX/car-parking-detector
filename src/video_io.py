"""
Video loading helpers for the dashboard.

`FrameReader` decodes AND downscales the next frames on a background thread. The demo clips
are 3840x2160 while the pipeline runs at 1280x720: decode + resize cost ~50-60 ms per frame,
which now overlaps with detection instead of adding to it. Frames are resized exactly as the
pipeline did inline (cv2.resize to 1280x720 when wider than 1280), so results are unchanged.

A pre-transcoded 720p copy was measured and rejected: Motion-JPEG is lossy (35.8-40 dB PSNR
vs the exact resize), takes 12-27 s per clip to build, and gains nothing once decoding runs
on this thread.
"""

import queue
import threading
from typing import Optional, Tuple

import cv2
import numpy as np

PIPELINE_SIZE = (1280, 720)


def fit_pipeline(frame: np.ndarray) -> np.ndarray:
    """The pipeline's input size rule: frames wider than 1280 px are resized to 1280x720."""
    return cv2.resize(frame, PIPELINE_SIZE) if frame.shape[1] > PIPELINE_SIZE[0] else frame


def read_frame_at(path: str, index: int) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


class FrameReader:
    """Iterates (frame, position_ms) from a video file, decoding + resizing ahead on a worker thread."""

    _END = object()

    def __init__(self, path: str, prefetch: int = 8):
        self.cap = cv2.VideoCapture(path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._q: "queue.Queue" = queue.Queue(maxsize=prefetch)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="frame-reader")
        self._thread.start()

    def isOpened(self) -> bool:
        return self.cap.isOpened()

    def _put(self, item) -> bool:
        while not self._stop.is_set():
            try:
                self._q.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _run(self):
        try:
            while not self._stop.is_set():
                ok, frame = self.cap.read()
                if not ok:
                    break
                pos_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
                if not self._put((fit_pipeline(frame), pos_ms)):
                    return
        finally:
            self._put(self._END)

    def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
        item = self._q.get()
        if item is self._END:
            self._q.put(item)  # keep returning end-of-stream on later calls
            return False, None, 0.0
        return True, item[0], item[1]

    def release(self):
        self._stop.set()
        self._thread.join(timeout=2.0)
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
