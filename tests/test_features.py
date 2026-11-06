import math

import numpy as np
import pytest

from src.trajectory_features import TrackHistory, TrackSample, pair_features, time_to_collision


def make_history(points, heading=0.0):
    h = TrackHistory()
    for t, x, y in points:
        h.push(TrackSample(t=t, x=x, y=y, heading=heading))
    return h


def test_head_on_approach_has_finite_decreasing_ttc():
    # Two cars 20 m apart, closing at 1 m/s each (2 m/s combined)
    a = make_history([(0.0, 0.0, 0.0), (0.1, 0.1, 0.0)])
    b = make_history([(0.0, 20.0, 0.0), (0.1, 19.9, 0.0)], heading=math.pi)
    feats = pair_features(a, b, fps=10)
    assert feats["ttc"] < 10.0
    assert feats["ttc"] > 0
    assert feats["v_close"] > 0  # closing


def test_parallel_motion_has_capped_ttc():
    # Two cars moving in the same direction, same speed, constant separation -> never meet
    a = make_history([(0.0, 0.0, 0.0), (0.1, 1.0, 0.0)])
    b = make_history([(0.0, 0.0, 5.0), (0.1, 1.0, 5.0)])
    feats = pair_features(a, b, fps=10)
    assert feats["ttc"] == pytest.approx(10.0)  # TTC_CAP_S, no convergence


def test_stationary_pair_has_capped_ttc():
    a = make_history([(0.0, 0.0, 0.0), (0.1, 0.0, 0.0)])
    b = make_history([(0.0, 10.0, 0.0), (0.1, 10.0, 0.0)])
    feats = pair_features(a, b, fps=10)
    assert feats["v_close"] == pytest.approx(0.0, abs=1e-6)
    assert feats["ttc"] == pytest.approx(10.0)


def test_time_to_collision_direct_formula():
    # Two discs 10 m apart, closing head-on at 2 m/s combined, radius 1.2 m each -> gap 7.6 m
    p_rel = np.array([10.0, 0.0])
    v_rel = np.array([-2.0, 0.0])
    ttc = time_to_collision(p_rel, v_rel, r=1.2)
    assert ttc == pytest.approx(7.6 / 2.0, rel=1e-3)


def test_reversing_detection():
    # heading points along +x, velocity points along -x -> reversing
    h = make_history([(0.0, 5.0, 0.0), (0.1, 4.9, 0.0)], heading=0.0)
    assert h.is_reversing() is True

    h2 = make_history([(0.0, 5.0, 0.0), (0.1, 5.1, 0.0)], heading=0.0)
    assert h2.is_reversing() is False
