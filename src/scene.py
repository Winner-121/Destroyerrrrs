"""Scene layout helpers for the fixed camera.

All polygons in `scene_layout.json` are in 1280x720 reference pixels drawn on
`reference_bg.jpg`. `Scene` first warps them with the per-video homography
(see align.py), then scales them to the working frame size, and answers
point-in-region queries, stop-line side tests and flow-direction lookups.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
LAYOUT_PATH = HERE / "scene_layout.json"
FLOW_PATH = HERE / "flow_field.npz"


class Scene:
    def __init__(self, width: int, height: int, H: np.ndarray | None = None,
                 layout_path: Path = LAYOUT_PATH, flow_path: Path = FLOW_PATH):
        L = json.loads(Path(layout_path).read_text())
        rw, rh = L["reference_size"]
        self.sx, self.sy = width / rw, height / rh
        self.width, self.height = width, height
        self.H = np.eye(3) if H is None else np.asarray(H, dtype=np.float64)

        def tf(pts):
            p = np.float32(pts).reshape(-1, 1, 2)
            p = cv2.perspectiveTransform(p, self.H.astype(np.float32)).reshape(-1, 2)
            p[:, 0] *= self.sx
            p[:, 1] *= self.sy
            return p.astype(np.float32)

        self.road = tf(L["road"])
        self.zebra_main = tf(L["zebra_main"])
        self.zebra_bottom = tf(L["zebra_bottom"])
        self.islands = [tf(v) for v in L["islands"].values()]
        self.approach = tf(L["near_carriageway_approach"])
        self.upstream = tf(L["near_carriageway_upstream"])
        self.zebra_side = tf(L["zebra_side"])
        self.side_mouth = tf(L["side_street_mouth"])
        self.box = tf(L["intersection_box"])
        self.parked = tf(L["parked_zone"])
        s = L["stop_line_near"]
        sp = tf([s["p1"], s["p2"]])
        self.stop_p1, self.stop_p2 = sp[0].astype(np.float64), sp[1].astype(np.float64)
        self.signals = {}
        for k in ("signal_main", "signal_left"):
            g = L[k]
            p = tf([g["red"], g["green"]])
            self.signals[k] = {
                "red": (int(round(p[0][0])), int(round(p[0][1]))),
                "green": (int(round(p[1][0])), int(round(p[1][1]))),
                "radius": max(2, int(round(g["radius"] * self.sx))),
            }

        # learned flow field (cell heading vectors) from the sample trajectories, reference coords
        f = np.load(flow_path)
        self.G = float(f["G"])
        cnt = np.maximum(f["cnt"], 1)
        self.ux, self.uy = f["vx"] / cnt, f["vy"] / cnt
        self.coh = np.hypot(self.ux, self.uy)
        self.cnt = f["cnt"]
        self.Hinv = np.linalg.inv(self.H)

    # ---- geometry -----------------------------------------------------------------
    @staticmethod
    def _inside(poly: np.ndarray, x: float, y: float) -> bool:
        return cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0

    def in_zebra(self, x, y) -> bool:
        return self._inside(self.zebra_main, x, y) or self._inside(self.zebra_bottom, x, y) or self._inside(self.zebra_side, x, y)

    def in_upstream(self, x, y) -> bool:
        return self._inside(self.upstream, x, y)

    def in_side_mouth(self, x, y) -> bool:
        return self._inside(self.side_mouth, x, y)

    def in_main_zebra(self, x, y) -> bool:
        return self._inside(self.zebra_main, x, y)

    def in_island(self, x, y) -> bool:
        return any(self._inside(p, x, y) for p in self.islands)

    def in_approach(self, x, y) -> bool:
        return self._inside(self.approach, x, y)

    def in_approach_or_zebra(self, x, y) -> bool:
        return self.in_approach(x, y) or self.in_main_zebra(x, y)

    def in_box(self, x, y) -> bool:
        return self._inside(self.box, x, y)

    def in_parked(self, x, y) -> bool:
        return self._inside(self.parked, x, y)

    def on_road(self, x, y) -> bool:
        """Carriageway proper: inside the road polygon, not on an island."""
        return self._inside(self.road, x, y) and not self.in_island(x, y)

    def road_margin(self, x, y) -> float:
        """Signed distance (working px) inside the carriageway: positive = inside road and
        at least that far from the kerb, islands and zebras. Negative = off the carriageway."""
        d = cv2.pointPolygonTest(self.road, (float(x), float(y)), True)
        for p in self.islands + [self.zebra_main, self.zebra_bottom, self.zebra_side]:
            dp = cv2.pointPolygonTest(p, (float(x), float(y)), True)
            if dp >= 0:
                return -abs(dp) - 1.0  # inside an island/zebra: not carriageway
            d = min(d, -dp)  # distance to that polygon's border
        return float(d)

    def past_stop_line(self, x, y) -> float:
        """Signed distance (working px) from the stop line; positive = past it (zebra side)."""
        d = self.stop_p2 - self.stop_p1
        n = np.array([d[1], -d[0]])
        n /= np.linalg.norm(n)
        dist = float((np.array([x, y]) - self.stop_p1) @ n)
        zc = self.zebra_main.mean(axis=0)
        if float((zc - self.stop_p1) @ n) < 0:
            dist = -dist
        return dist

    def to_reference(self, x, y):
        """Working-frame pixel -> reference-frame pixel (undo scale and homography)."""
        p = np.array([x / self.sx, y / self.sy, 1.0])
        q = self.Hinv @ p
        return q[0] / q[2], q[1] / q[2]

    def flow_at(self, x, y):
        """(unit heading vector, coherence, count) of normal traffic at working pixel (x, y)."""
        rx, ry = self.to_reference(x, y)
        gy, gx = int(ry // self.G), int(rx // self.G)
        if gy < 0 or gx < 0 or gy >= self.ux.shape[0] or gx >= self.ux.shape[1]:
            return np.zeros(2), 0.0, 0
        c = self.coh[gy, gx]
        u = np.array([self.ux[gy, gx], self.uy[gy, gx]]) / c if c > 1e-6 else np.zeros(2)
        return u, float(c), int(self.cnt[gy, gx])
