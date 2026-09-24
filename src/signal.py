"""Traffic-signal state from lamp pixels of the two visible signal heads.

Both heads control the near carriageway and run in the same phase. The
far-pole head is bright at dusk but washes out in direct sun; the near-pole
head is reliable in daylight. We score every lamp per frame (red = R minus the
mean of G and B; green = mean of G and B minus R, because the green lamp is
cyan on this camera), then decide the state either

  * offline (Part A): thresholds adapted per lamp from the whole video, or
  * online  (Part B): fixed thresholds, causal hysteresis.
"""
from __future__ import annotations

import numpy as np

RED, GREEN, AMBER, UNKNOWN = "red", "green", "amber", "unknown"


def lamp_scores(frame: np.ndarray, scene) -> np.ndarray:
    """[red_main, green_main, red_left, green_left] for one BGR frame."""
    out = []
    for k in ("signal_main", "signal_left"):
        g = scene.signals[k]
        r = g["radius"]
        for which, kind in (("red", "r"), ("green", "g")):
            x, y = g[which]
            y0, y1 = max(0, y - r), min(frame.shape[0], y + r + 1)
            x0, x1 = max(0, x - r), min(frame.shape[1], x + r + 1)
            p = frame[y0:y1, x0:x1].astype(np.float32)
            if p.size == 0:
                out.append(0.0)
                continue
            b, gg, rr = p[..., 0], p[..., 1], p[..., 2]
            out.append(float(np.mean(rr - (gg + b) / 2)) if kind == "r" else float(np.mean((gg + b) / 2 - rr)))
    return np.array(out, dtype=np.float32)


def _decide(red_lit: bool, grn_lit: bool, prev: str) -> str:
    if red_lit and not grn_lit:
        return RED
    if grn_lit and not red_lit:
        return GREEN
    if red_lit and grn_lit:
        return UNKNOWN
    return AMBER if prev in (GREEN, AMBER) else UNKNOWN  # dark after green = amber phase


def states_offline(scores: np.ndarray, min_thr: float = 10.0, frac: float = 0.45, hold: int = 2) -> list[str]:
    """Per-sample state from an (N,4) score matrix, thresholds adapted per lamp.

    A lamp is 'lit' when its score exceeds frac * its own 95th percentile (and
    min_thr). Heads are combined by OR on red and OR on green. Unknown samples
    inherit the previous state; a switch needs `hold` consistent samples.
    """
    S = np.asarray(scores, dtype=np.float32)
    p95 = np.percentile(S, 95, axis=0)
    thr = np.maximum(min_thr, frac * p95)
    lit = S > thr
    # a head whose lamps never light (p95 tiny) is ignored
    usable = p95 > min_thr
    red_lit = (lit[:, 0] & usable[0]) | (lit[:, 2] & usable[2])
    grn_lit = (lit[:, 1] & usable[1]) | (lit[:, 3] & usable[3])
    states, prev, pend, n = [], UNKNOWN, None, 0
    for r, g in zip(red_lit, grn_lit):
        raw = _decide(bool(r), bool(g), prev)
        if raw == UNKNOWN or raw == prev:
            pend, n = None, 0
        elif raw == pend:
            n += 1
        else:
            pend, n = raw, 1
        if pend is not None and (n >= hold or prev == UNKNOWN):
            prev, pend, n = pend, None, 0
        states.append(prev)
    # back-fill leading unknowns with the first known state
    first = next((s for s in states if s != UNKNOWN), UNKNOWN)
    return [first if s == UNKNOWN else s for s in states]


class OnlineSignal:
    """Causal reader for Part B: fixed thresholds + hysteresis."""

    def __init__(self, scene, thr: float = 14.0, hold: int = 2):
        self.scene, self.thr, self.hold = scene, thr, hold
        self.state, self._pend, self._n = UNKNOWN, None, 0

    def update(self, frame: np.ndarray) -> str:
        s = lamp_scores(frame, self.scene)
        red = (s[0] > self.thr) or (s[2] > self.thr)
        grn = (s[1] > self.thr) or (s[3] > self.thr)
        raw = _decide(red, grn, self.state)
        if raw == UNKNOWN or raw == self.state:
            self._pend, self._n = None, 0
            return self.state
        if raw == self._pend:
            self._n += 1
        else:
            self._pend, self._n = raw, 1
        if self._n >= self.hold or self.state == UNKNOWN:
            self.state, self._pend, self._n = raw, None, 0
        return self.state
