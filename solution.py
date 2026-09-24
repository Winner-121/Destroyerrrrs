"""
solution.py — WIUT Hackathon 2026, CV track.

Part A: detect_events(video_path) -> [[start_sec, end_sec, label], ...]
Part B: RiskEstimator (causal accident anticipation)

Pipeline (see README): YOLO11m + ByteTrack on every 3rd frame at 1280 px,
per-video alignment of a hand-drawn scene layout (homography from the median
background), traffic-signal state from the two visible signal heads, and
rule-based event logic on the resulting trajectories.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Classes we actually predict. (Removing ids we never output is allowed; adding is not.)
CLASSES: list[str] = [
    "red_light",
    "stopped_vehicle",
    "jaywalking",
    "failure_to_yield",
    "stop_line",
    "wrong_way",
]

RISK_HORIZON_SEC = 5.0


def detect_events(video_path: str) -> list[list]:
    """Part A — traffic event detection."""
    from src.pipeline import detect_events as _detect
    events = _detect(video_path, log=lambda *a, **k: print(*a, **k, flush=True))
    return [[float(s), float(e), str(l)] for s, e, l in events if l in CLASSES]


class RiskEstimator:
    """Part B — causal accident anticipation (placeholder: returns 0)."""

    def reset(self, meta: dict) -> None:
        self.meta = meta
        self.last_score = 0.0

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        return self.last_score
