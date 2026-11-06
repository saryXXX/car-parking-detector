"""
Pairwise conflict features for the learned collision / near-miss model (add-on F1).

All distances in metres, speeds in m/s, times in seconds. Pure numpy: no dependency on
the tracker or spatial engine, so it is unit-testable in isolation and reusable both for
offline training (on simulated trajectories) and at runtime (on real tracked positions).
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np

CAR_DISC_RADIUS_M = 1.2  # approximates each car as a disc of this radius for TTC
TTC_CAP_S = 10.0


@dataclass
class TrackSample:
    """One car's state at one instant."""
    t: float
    x: float
    y: float
    heading: float  # radians, direction of travel when moving forward


@dataclass
class TrackHistory:
    """Up to the last ~1s of samples for one car, oldest first."""
    samples: List[TrackSample] = field(default_factory=list)

    def push(self, sample: TrackSample, window_s: float = 1.0):
        self.samples.append(sample)
        cutoff = sample.t - window_s
        self.samples = [s for s in self.samples if s.t >= cutoff]

    @property
    def latest(self) -> TrackSample:
        return self.samples[-1]

    def velocity(self) -> np.ndarray:
        """Finite-difference velocity from the two most recent samples."""
        if len(self.samples) < 2:
            return np.zeros(2)
        a, b = self.samples[-2], self.samples[-1]
        dt = b.t - a.t
        if dt <= 0:
            return np.zeros(2)
        return np.array([(b.x - a.x) / dt, (b.y - a.y) / dt])

    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity()))

    def max_decel_last(self, window_s: float = 0.5) -> float:
        """Max braking (speed drop per second) over the given trailing window."""
        pts = [s for s in self.samples if s.t >= self.latest.t - window_s]
        if len(pts) < 3:
            return 0.0
        speeds = []
        for i in range(1, len(pts)):
            dt = pts[i].t - pts[i - 1].t
            if dt <= 0:
                continue
            v = np.hypot(pts[i].x - pts[i - 1].x, pts[i].y - pts[i - 1].y) / dt
            speeds.append((pts[i].t, v))
        if len(speeds) < 2:
            return 0.0
        decel = 0.0
        for i in range(1, len(speeds)):
            dt = speeds[i][0] - speeds[i - 1][0]
            if dt <= 0:
                continue
            decel = max(decel, (speeds[i - 1][1] - speeds[i][1]) / dt)
        return decel

    def is_reversing(self) -> bool:
        """True if velocity points opposite to the car's heading (reversing)."""
        v = self.velocity()
        if np.linalg.norm(v) < 1e-3:
            return False
        h = np.array([np.cos(self.latest.heading), np.sin(self.latest.heading)])
        return float(np.dot(v, h)) < 0


def time_to_collision(p_rel: np.ndarray, v_rel: np.ndarray, r: float = CAR_DISC_RADIUS_M) -> float:
    """TTC for two discs of radius r, relative position p_rel and velocity v_rel.
    Returns TTC_CAP_S if no future collision under constant velocity."""
    a = float(v_rel @ v_rel)
    if a < 1e-9:
        return TTC_CAP_S
    b = 2 * float(p_rel @ v_rel)
    c = float(p_rel @ p_rel) - (2 * r) ** 2
    disc = b * b - 4 * a * c
    if disc < 0:
        return TTC_CAP_S
    t = (-b - np.sqrt(disc)) / (2 * a)
    return min(t, TTC_CAP_S) if t > 0 else TTC_CAP_S


def pair_features(hist_a: TrackHistory, hist_b: TrackHistory, fps: float,
                   aisle_polygon=None) -> dict:
    """Compute the F1 feature set for one pair of tracks at the current instant."""
    a, b = hist_a.latest, hist_b.latest
    p_rel = np.array([b.x - a.x, b.y - a.y])
    v_a, v_b = hist_a.velocity(), hist_b.velocity()
    v_rel = v_b - v_a

    d = float(np.linalg.norm(p_rel)) - 2 * CAR_DISC_RADIUS_M
    d = max(0.0, d)
    v_close = -float(np.dot(p_rel, v_rel)) / (np.linalg.norm(p_rel) + 1e-9)
    ttc = time_to_collision(p_rel, v_rel)

    # ttc_min_1s: recompute TTC at each sample pair in the shared trailing window
    ttc_hist = []
    n = min(len(hist_a.samples), len(hist_b.samples))
    for sa, sb in zip(hist_a.samples[-n:], hist_b.samples[-n:]):
        pr = np.array([sb.x - sa.x, sb.y - sa.y])
        ttc_hist.append(time_to_collision(pr, v_rel))
    ttc_min_1s = min(ttc_hist) if ttc_hist else ttc

    speed_a, speed_b = hist_a.speed(), hist_b.speed()
    heading_rel = float(np.arctan2(np.sin(a.heading - b.heading), np.cos(a.heading - b.heading)))

    def in_polygon(pt):
        if aisle_polygon is None:
            return True
        from shapely.geometry import Point, Polygon
        return Polygon(aisle_polygon).contains(Point(pt))

    return {
        "d": d,
        "v_close": v_close,
        "ttc": ttc,
        "ttc_min_1s": ttc_min_1s,
        "heading_rel": heading_rel,
        "speed_a": speed_a,
        "speed_b": speed_b,
        "decel_a": hist_a.max_decel_last(),
        "decel_b": hist_b.max_decel_last(),
        "reversing_a": float(hist_a.is_reversing()),
        "reversing_b": float(hist_b.is_reversing()),
        "in_aisle_a": float(in_polygon((a.x, a.y))),
        "in_aisle_b": float(in_polygon((b.x, b.y))),
    }


FEATURE_NAMES = [
    "d", "v_close", "ttc", "ttc_min_1s", "heading_rel", "speed_a", "speed_b",
    "decel_a", "decel_b", "reversing_a", "reversing_b", "in_aisle_a", "in_aisle_b",
]
