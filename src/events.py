"""Rules: tracks + scene layout + signal state  ->  per-frame flags  ->  event segments.

Input tracks are rows [t, id, cls, conf, x1, y1, x2, y2] in working-frame pixels,
sampled every `stride` frames. COCO classes: 0 person, 1 bicycle, 2 car,
3 motorcycle, 5 bus, 7 truck.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

PERSON, BICYCLE, CAR, MOTORCYCLE, BUS, TRUCK = 0, 1, 2, 3, 5, 7
VEHICLES = {CAR, MOTORCYCLE, BUS, TRUCK}


# ----------------------------------------------------------------------------- helpers
def group_tracks(rows):
    """-> {id: (t[], cls, cx[], cy_bottom[], w[], h[], conf[])} sorted by time."""
    tr = defaultdict(list)
    for t, i, c, cf, x1, y1, x2, y2 in rows:
        tr[int(i)].append((float(t), int(c), (x1 + x2) / 2, float(y2), x2 - x1, y2 - y1, float(cf)))
    out = {}
    for i, p in tr.items():
        p.sort()
        a = np.array(p, dtype=np.float64)
        cls = int(np.bincount(a[:, 1].astype(int)).argmax())  # majority class
        out[i] = dict(t=a[:, 0], cls=cls, cx=a[:, 2], cy=a[:, 3], w=a[:, 4], h=a[:, 5], conf=a[:, 6])
    return out


def flags_to_segments(times, flags, min_dur=0.5, max_gap=1.0):
    """Boolean flags at sample times -> [(start, end)], gaps <= max_gap merged, short ones dropped."""
    segs = []
    start = None
    last_true = None
    for t, f in zip(times, flags):
        if f:
            if start is None:
                start = t
            elif last_true is not None and t - last_true > max_gap:
                segs.append((start, last_true))
                start = t
            last_true = t
        # a False sample does not close the run; the gap rule does
    if start is not None:
        segs.append((start, last_true))
    # merge again after per-track union
    return [(s, e) for s, e in segs if e - s >= min_dur]


def merge_intervals(segs, gap=0.0):
    """Union of intervals (same class), merging ones closer than `gap`."""
    segs = sorted(segs)
    out = []
    for s, e in segs:
        if out and s <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def speed(track, k, win_s=1.0):
    """Displacement speed (px/s) of the ground point around sample k."""
    t, cx, cy = track["t"], track["cx"], track["cy"]
    j = k
    while j + 1 < len(t) and t[j + 1] - t[k] <= win_s:
        j += 1
    i = k
    while i - 1 >= 0 and t[k] - t[i - 1] <= win_s:
        i -= 1
    dt = t[j] - t[i]
    if dt <= 0:
        return 0.0
    return float(np.hypot(cx[j] - cx[i], cy[j] - cy[i]) / dt)


# ----------------------------------------------------------------------------- rules
class RuleEngine:
    def __init__(self, scene, fps: float, stride: int):
        self.sc = scene
        self.fps = fps
        self.dt = stride / fps
        # thresholds in reference-720p pixels, scaled to the working frame
        s = scene.sx
        self.still_px_s = 6.0 * s          # below this ground speed a vehicle is 'stopped'
        self.min_track_s = 1.0

    # --- jaywalking: pedestrian on the carriageway outside any crossing / island -----
    def jaywalking(self, tracks, signal_by_t=None):
        # vehicle boxes per time, to drop 'person' detections that are riders / passengers
        veh_at = defaultdict(list)
        for tr in tracks.values():
            if tr["cls"] in VEHICLES or tr["cls"] == BICYCLE:
                for t, x, yb, w, h in zip(tr["t"], tr["cx"], tr["cy"], tr["w"], tr["h"]):
                    veh_at[round(t, 2)].append((x - w / 2, yb - h, x + w / 2, yb))
        per_track = []
        for tr in tracks.values():
            if tr["cls"] != PERSON or tr["t"][-1] - tr["t"][0] < self.min_track_s:
                continue
            fl = []
            for t, x, y, w, h in zip(tr["t"], tr["cx"], tr["cy"], tr["w"], tr["h"]):
                on = self.sc.road_margin(x, y) > 14 * self.sc.sx and not self.sc.in_parked(x, y)
                if on:  # inside a vehicle/bicycle box -> rider or passenger, not a pedestrian
                    cy = y - h / 2
                    for x1, y1, x2, y2 in veh_at.get(round(t, 2), ()):
                        if x1 <= x <= x2 and y1 <= cy <= y2 and (x2 - x1) < 4 * w:
                            on = False
                            break
                fl.append(on)
            for s_, e_ in flags_to_segments(tr["t"], fl, min_dur=2.0, max_gap=1.0):
                k0, k1 = int(np.searchsorted(tr["t"], s_)), int(np.searchsorted(tr["t"], e_))
                path = float(np.hypot(np.diff(tr["cx"][k0:k1 + 1]), np.diff(tr["cy"][k0:k1 + 1])).sum())
                if path >= 40 * self.sc.sx:  # a pedestrian who actually walks, not a static false detection
                    per_track.append((s_, e_))
        return merge_intervals(per_track, gap=1.0)

    # --- stopped vehicle: stationary >= 10 s on the road, not queued at a signal -----
    def stopped_vehicle(self, tracks, signal_by_t):
        """Queue handling: a vehicle stopped upstream of the near stop line while the
        near signal is not green (or within 20 s after it turned green) is a queue.
        Elsewhere on the road, other directions may also queue for signals we cannot
        see, so outside the intersection box we require >= 60 s (longer than a full
        red phase); inside the box / on a zebra 10 s is enough."""
        segs = []
        green_starts = self._phase_starts(signal_by_t, "green")
        for tr in tracks.values():
            if tr["cls"] not in VEHICLES or tr["t"][-1] - tr["t"][0] < 10:
                continue
            n = len(tr["t"])
            still = np.array([speed(tr, k) < self.still_px_s for k in range(n)])
            xs, ys, ts = tr["cx"], tr["cy"], tr["t"]
            onroad = np.array([self.sc.on_road(x, y) and not self.sc.in_parked(x, y) for x, y in zip(xs, ys)])
            inbox = np.array([(self.sc.in_box(x, y) or self.sc.in_main_zebra(x, y)) and not self.sc.in_side_mouth(x, y) for x, y in zip(xs, ys)])
            upstream = np.array([self.sc.in_upstream(x, y) and self.sc.past_stop_line(x, y) < 0 for x, y in zip(xs, ys)])
            queued = np.array([u and (signal_by_t(t) != "green" or self._since(green_starts, t) < 20) for u, t in zip(upstream, ts)])
            fl = still & onroad & ~queued
            moved_any = (~still).any()
            for s_, e_ in flags_to_segments(ts, fl, min_dur=10.0, max_gap=1.5):
                k0 = int(np.searchsorted(ts, s_)); k1 = int(np.searchsorted(ts, e_))
                frac_box = inbox[k0:k1 + 1].mean() if k1 >= k0 else 0
                need = 10.0 if frac_box > 0.5 else 60.0
                # a vehicle that never moves in the whole clip outside the intersection is parked
                # scenery (kerb lane at the bus stop), not an event we can bound in time
                if frac_box <= 0.5 and not moved_any:
                    continue
                if e_ - s_ >= need:
                    segs.append((s_, e_))
        return merge_intervals(segs, gap=0.5)

    @staticmethod
    def _phase_starts(signal_by_t, state, t_max=7200, step=0.1):
        starts, prev, t = [], None, 0.0
        while t < t_max:
            s = signal_by_t(t)
            if s == state and prev != state:
                starts.append(t)
            prev = s
            t += step
            if t > 5 and s == "unknown" and prev == "unknown" and not starts and t > 600:
                break
        return np.array(starts)

    @staticmethod
    def _since(starts, t):
        if len(starts) == 0:
            return 1e9
        k = int(np.searchsorted(starts, t)) - 1
        return t - starts[k] if k >= 0 else 1e9

    # --- stop line: vehicle stopped past the stop line (front on the zebra) on red -----
    def stop_line(self, tracks, signal_by_t):
        segs = []
        for tr in tracks.values():
            if tr["cls"] not in VEHICLES:
                continue
            fl = []
            for k in range(len(tr["t"])):
                x, y, t = tr["cx"][k], tr["cy"][k], tr["t"][k]
                d = self.sc.past_stop_line(x, y)
                past = 8 * self.sc.sx < d < 60 * self.sc.sx  # clearly over the bar, not through the zebra
                fl.append(past and speed(tr, k) < self.still_px_s and signal_by_t(t) == "red" and self.sc.in_approach_or_zebra(x, y))
            for s_, e_ in flags_to_segments(tr["t"], fl, min_dur=3.0, max_gap=1.5):
                segs.append((s_, e_))
        return merge_intervals(segs, gap=1.0)

    # --- red light: crosses the stop line on red and clears the zebra promptly -----
    def red_light(self, tracks, signal_by_t):
        segs = []
        far = 60 * self.sc.sx
        for tr in tracks.values():
            if tr["cls"] not in VEHICLES:
                continue
            xs, ys, ts = tr["cx"], tr["cy"], tr["t"]
            d = np.array([self.sc.past_stop_line(x, y) for x, y in zip(xs, ys)])
            inapp = np.array([self.sc.in_approach(x, y) for x, y in zip(xs, ys)])
            for k in range(1, len(d)):
                if not (inapp[k - 1] and d[k - 1] <= 0 < d[k]):
                    continue
                t = ts[k]
                # provenance: came down the near carriageway (upstream polygon) for >= 2 s
                # before the stop line, heading with the flow; turning traffic from the
                # median gap / side street is governed by other signals
                k0 = k - 1
                while k0 > 0 and ts[k] - ts[k0] < 2.0:
                    k0 -= 1
                if ts[k] - ts[k0] < 1.5 or not all(self.sc.in_upstream(xs[j], ys[j]) for j in range(k0, k)):
                    continue
                dx, dy = xs[k] - xs[k0], ys[k] - ys[k0]
                L = float(np.hypot(dx, dy)) + 1e-6
                u, coh, cnt = self.sc.flow_at(xs[k0], ys[k0])
                if coh > 0.5 and float(u @ np.array([dx, dy]) / L) < 0.6:
                    continue
                # the near-pole head goes red ~6 s before the far-pole head (it skips the amber
                # phase), so 'red' for vehicles starts 6 s after the combined red state begins
                if not (signal_by_t(t) == "red" and signal_by_t(t - 6.0) == "red" and signal_by_t(t + 1.0) == "red"):
                    continue
                if speed(tr, k) < 2 * self.still_px_s:
                    continue
                later = np.where(d[k:] > far)[0]
                if len(later) == 0 or ts[k + later[0]] - t > 4.0:
                    continue  # crept onto the zebra and waited: that is stop_line, not red_light
                j = k + later[0]
                end = min(ts[-1], ts[j] + 1.5)
                segs.append((t, end))
                break
        return merge_intervals(segs, gap=0.5)

    # --- failure to yield: vehicle crosses the main zebra near a pedestrian who is on it -----
    def failure_to_yield(self, tracks, signal_by_t):
        """Vehicle moving across the main zebra while a pedestrian is on the zebra within
        `near` px of it (the literal class definition; the crossing is ~900 px long)."""
        peds = defaultdict(list)
        for tr in tracks.values():
            if tr["cls"] != PERSON:
                continue
            for t, x, y in zip(tr["t"], tr["cx"], tr["cy"]):
                if self.sc.in_main_zebra(x, y) and self.sc.on_road(x, y):
                    peds[round(t, 2)].append((x, y))
        near = 220 * self.sc.sx
        segs = []
        for tr in tracks.values():
            if tr["cls"] not in VEHICLES:
                continue
            fl = []
            for k, (t, x, y) in enumerate(zip(tr["t"], tr["cx"], tr["cy"])):
                ok = self.sc.in_main_zebra(x, y) and speed(tr, k) > 1.5 * self.still_px_s
                if ok:
                    ok = any(np.hypot(px - x, py - y) < near for px, py in peds.get(round(t, 2), ()))
                fl.append(ok)
            segs += flags_to_segments(tr["t"], fl, min_dur=0.6, max_gap=1.0)
        return merge_intervals(segs, gap=1.0)

    # --- wrong way: heading opposite the learned flow for > 3 s, outside the intersection box -----
    def wrong_way(self, tracks, signal_by_t):
        segs = []
        for tr in tracks.values():
            if tr["cls"] not in VEHICLES or len(tr["t"]) < 8:
                continue
            fl = []
            for k in range(len(tr["t"])):
                j = min(len(tr["t"]) - 1, k + max(1, int(1.0 / self.dt)))
                dx, dy = tr["cx"][j] - tr["cx"][k], tr["cy"][j] - tr["cy"][k]
                L = float(np.hypot(dx, dy))
                x, y = tr["cx"][k], tr["cy"][k]
                if L < 2 * self.still_px_s or self.sc.in_box(x, y) or not self.sc.on_road(x, y):
                    fl.append(False)
                    continue
                u, coh, cnt = self.sc.flow_at(x, y)
                fl.append(coh > 0.85 and cnt > 60 and float(u @ np.array([dx, dy]) / L) < -0.9)
            segs += flags_to_segments(tr["t"], fl, min_dur=1.0, max_gap=0.7)
        return merge_intervals(segs, gap=1.0)

    # --- solid line crossing: ground point switches side of a solid divider while moving -----
    def solid_line_crossing(self, tracks, signal_by_t):
        """Ground point drifts from one side of a solid divider to the other: within 1.5 s
        before the sign flip it was clearly (> thr) on one side and never on the other,
        within 1.5 s after it is clearly on the other side and never back.
        Segment: from the farthest old-side sample before the flip (the lane change
        begins) to 1.5 s after the vehicle is clearly in the new lane."""
        segs = []
        w = max(3, int(1.5 / self.dt))
        thr = 6.0 * self.sc.sx
        tol = 2.0 * self.sc.sx
        for tr in tracks.values():
            if tr["cls"] not in VEHICLES or len(tr["t"]) < 6:
                continue
            xs, ys, ts = tr["cx"], tr["cy"], tr["t"]
            sides = [self.sc.solid_line_side(x, y) for x, y in zip(xs, ys)]
            for li in range(len(self.sc.solid_lines)):
                sd = np.array([np.nan if s_[li] is None else s_[li] for s_ in sides], dtype=np.float64)
                k = 1
                while k < len(sd) - 3:
                    a_, b_ = sd[k - 1], sd[k]
                    if np.isnan(a_) or np.isnan(b_) or (a_ < 0) == (b_ < 0):
                        k += 1
                        continue
                    lo = max(0, k - w)
                    before, after = sd[lo:k], sd[k:k + w]
                    bef, aft = before[~np.isnan(before)], after[~np.isnan(after)]
                    if len(bef) < 3 or len(aft) < 3:
                        k += 1
                        continue
                    sign = 1.0 if b_ > 0 else -1.0  # direction of the flip
                    ok = ((sign * bef).min() < -thr and (sign * bef <= tol).all()
                          and (sign * aft).max() > thr and (sign * aft >= -tol).all())
                    if ok and speed(tr, k) > self.still_px_s and (self.sc.in_upstream(xs[k], ys[k]) or self.sc.in_upstream(xs[lo], ys[lo])):
                        vals = np.where(np.isnan(before), np.inf, sign * before)
                        i0 = lo + int(np.argmin(vals))                    # farthest on the old side
                        i1 = k
                        while i1 < min(len(sd) - 1, k + w - 1) and not (not np.isnan(sd[i1]) and sign * sd[i1] > thr):
                            i1 += 1
                        segs.append((max(ts[0], ts[i0]), min(ts[-1], ts[i1] + 1.5)))
                        k += w
                        continue
                    k += 1
        return merge_intervals(segs, gap=0.5)

    # --- illegal turn: a moving vehicle drives over a pedestrian / traffic island -----
    def illegal_turn(self, tracks, signal_by_t):
        """A vehicle drives over a pedestrian / traffic island: its ground-contact strip lies
        largely on the island, its ground point is on (or at) the island, and it moves at
        mounting speed (fast traffic passing behind an island in image space is excluded)."""
        segs = []
        v_max = 100.0 * self.sc.sx
        for tr in tracks.values():
            if tr["cls"] not in VEHICLES or len(tr["t"]) < 3:
                continue
            fl = []
            for k, (x, yb, w, h) in enumerate(zip(tr["cx"], tr["cy"], tr["w"], tr["h"])):
                m = self.sc.island_margin(x, yb)
                if m < -4 * self.sc.sx:
                    fl.append(False)
                    continue
                cov = self.sc.island_cover(x - w / 2, yb - h, x + w / 2, yb, frac_h=0.1)
                v = speed(tr, k)
                fl.append(cov > 0.3 and self.still_px_s < v < v_max)
            for s_, e_ in flags_to_segments(tr["t"], fl, min_dur=0.0, max_gap=1.0):
                segs.append((max(tr["t"][0], s_ - 1.0), min(tr["t"][-1], e_ + 1.5)))
        return merge_intervals(segs, gap=1.0)

    # --- congestion: the intersection box itself is jammed with stationary vehicles -----
    def congestion(self, tracks, signal_by_t, min_vehicles=4, min_dur=5.0):
        """Queues spilling into / blocking the intersection: >= min_vehicles distinct vehicles
        stationary inside the box (not on the zebra) at the same time for >= min_dur."""
        per_t = defaultdict(set)
        for i, tr in tracks.items():
            if tr["cls"] not in VEHICLES:
                continue
            for k, (t, x, y) in enumerate(zip(tr["t"], tr["cx"], tr["cy"])):
                if self.sc.in_box(x, y) and not self.sc.in_main_zebra(x, y) and not self.sc.in_side_mouth(x, y) \
                        and speed(tr, k) < self.still_px_s:
                    per_t[round(t, 2)].add(i)
        ts = sorted(per_t)
        fl = [len(per_t[t]) >= min_vehicles for t in ts]
        return flags_to_segments(np.array(ts), fl, min_dur=min_dur, max_gap=3.0)

    # --- illegal U-turn: heading reverses (> 150 deg) within a few seconds while moving -----
    def illegal_u_turn(self, tracks, signal_by_t):
        segs = []
        w = max(2, int(1.0 / self.dt))
        for tr in tracks.values():
            if tr["cls"] not in VEHICLES or len(tr["t"]) < 4 * w:
                continue
            xs, ys, ts = tr["cx"], tr["cy"], tr["t"]
            k = w
            while k < len(ts) - w:
                h0 = np.array([xs[k] - xs[k - w], ys[k] - ys[k - w]])
                L0 = float(np.linalg.norm(h0))
                if L0 < 25 * self.sc.sx:
                    k += 1
                    continue
                # look ahead up to 6 s for a reversed heading
                j = k + w
                hit = None
                while j < len(ts) and ts[j] - ts[k] <= 6.0:
                    h1 = np.array([xs[j] - xs[j - w], ys[j] - ys[j - w]])
                    L1 = float(np.linalg.norm(h1))
                    if L1 >= 25 * self.sc.sx and float(h0 @ h1) / (L0 * L1) < -0.87:
                        hit = j
                        break
                    j += 1
                if hit is not None and self.sc.on_road(xs[k], ys[k]):
                    segs.append((ts[k - w], min(ts[-1], ts[hit] + 1.0)))
                    k = hit + w
                    continue
                k += 1
        return merge_intervals(segs, gap=1.0)

    def run(self, rows, signal_by_t, classes=None):
        tracks = group_tracks(rows)
        rules = {
            "jaywalking": self.jaywalking,
            "stopped_vehicle": self.stopped_vehicle,
            "stop_line": self.stop_line,
            "red_light": self.red_light,
            "failure_to_yield": self.failure_to_yield,
            "wrong_way": self.wrong_way,
            "solid_line_crossing": self.solid_line_crossing,
            "congestion": self.congestion,
        }
        events = []
        for name, fn in rules.items():
            if classes and name not in classes:
                continue
            for s, e in fn(tracks, signal_by_t):
                events.append([round(float(s), 2), round(float(e), 2), name])
        return sorted(events)
