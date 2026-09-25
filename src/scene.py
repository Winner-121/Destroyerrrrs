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
        _za = (self.zebra_main[1] + self.zebra_main[2]) / 2 - (self.zebra_main[0] + self.zebra_main[3]) / 2
        self.zebra_axis = tuple((_za / (np.linalg.norm(_za) + 1e-6)).tolist())  # unit vector along the crossing
        self.zebra_bottom = tf(L["zebra_bottom"])
        self.islands = [tf(v) for v in L["islands"].values()]
        self.approach = tf(L["near_carriageway_approach"])
        self.upstream = tf(L["near_carriageway_upstream"])
        self.zebra_side = tf(L["zebra_side"])
        self.side_mouth = tf(L["side_street_mouth"])
        self.solid_lines = [tf(seg).astype(np.float64) for seg in L.get("solid_lines", [])]
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

    def island_cover(self, x1, y1, x2, y2, frac_h: float = 0.15) -> float:
        """Share of the vehicle's ground-contact strip (lower `frac_h` of the box) that lies on
        a small island (not the median). ~0 for a vehicle passing beside an island, high when
        its wheels are on it. Big boxes (buses) beside an island stay near 0 because the strip
        is thin."""
        best = 0.0
        y0 = y2 - frac_h * (y2 - y1)
        strip_area = max(1.0, (x2 - x1) * (y2 - y0))
        for p in self.islands[1:]:
            px0, py0 = p.min(axis=0)
            px1, py1 = p.max(axis=0)
            if x2 < px0 or x1 > px1 or y2 < py0 or y0 > py1:
                continue
            ox0, oy0 = int(min(px0, x1)), int(min(py0, y0))
            w, h = int(max(px1, x2) - ox0) + 2, int(max(py1, y2) - oy0) + 2
            isl = np.zeros((h, w), np.uint8)
            cv2.fillPoly(isl, [np.round(p - [ox0, oy0]).astype(np.int32)], 1)
            bx = np.zeros_like(isl)
            cv2.rectangle(bx, (int(x1 - ox0), int(y0 - oy0)), (int(x2 - ox0), int(y2 - oy0)), 1, -1)
            best = max(best, float((isl & bx).sum()) / strip_area)
        return best

    def island_margin(self, x, y) -> float:
        """Largest signed distance to any island polygon (positive = inside)."""
        return max(cv2.pointPolygonTest(p, (float(x), float(y)), True) for p in self.islands)

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

    def solid_line_side(self, x, y):
        """For each solid line: signed side of (x, y) if its projection lies within the segment
        (with a small margin), else None."""
        out = []
        for seg in self.solid_lines:
            p1, p2 = seg[0], seg[1]
            d = p2 - p1
            L = float(np.linalg.norm(d))
            u = d / L
            v = np.array([x, y], dtype=np.float64) - p1
            along = float(v @ u)
            if -0.05 * L <= along <= 1.6 * L:  # extended past the stop line so the 'after' side is observable
                n = np.array([u[1], -u[0]])
                out.append(float(v @ n))
            else:
                out.append(None)
        return out

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
