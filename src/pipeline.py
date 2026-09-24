"""End-to-end Part A pipeline for one video.

    frames (every `stride`-th, 4K -> 1280 px)  ->  YOLO11 + ByteTrack  ->  tracks
    sampled frames -> median background -> homography to the reference layout
    sampled frames -> signal lamp scores -> red / green / amber per sample
    tracks + aligned layout + signal -> rule engine -> [[start, end, label], ...]
"""
from __future__ import annotations

import os
import random
import time
from pathlib import Path

import cv2
import numpy as np

from .align import REF_W, REF_H, estimate_homography, describe
from .events import RuleEngine
from .scene import Scene
from .signal import lamp_scores, states_offline

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WEIGHTS = ROOT / "weights" / "yolo11m.pt"
REF_BG = HERE / "reference_bg.jpg"

DET_CLASSES = [0, 1, 2, 3, 5, 7]  # person, bicycle, car, motorcycle, bus, truck
SEED = 0


def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        from ultralytics import YOLO
        _MODEL = YOLO(str(WEIGHTS))
    return _MODEL


def device_and_half():
    try:
        import torch
        if torch.cuda.is_available():
            return 0, True
    except Exception:
        pass
    return "cpu", False


def detect_events(video_path: str, stride: int = 3, imgsz: int = 1280, log=print, return_debug: bool = False):
    set_seeds()
    t_start = time.perf_counter()
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    sx, sy = REF_W / W, REF_H / H
    duration = n_frames / fps

    model = get_model()
    device, half = device_and_half()
    n_samples = max(1, n_frames // stride)
    bg_every = max(1, n_samples // 25)

    # pass 1: detect + track + collect background samples and lamp crops
    rows, bg_frames, lamp_frames = [], [], []
    n = 0
    scene0 = Scene(REF_W, REF_H)  # unaligned, only to know roughly where the lamps are
    for r in model.track(source=video_path, stream=True, imgsz=imgsz, conf=0.25, classes=DET_CLASSES,
                         tracker="bytetrack.yaml", persist=True, verbose=False, vid_stride=stride,
                         half=half, device=device):
        t = (n * stride) / fps
        small = cv2.resize(r.orig_img, (REF_W, REF_H), interpolation=cv2.INTER_AREA)
        if n % bg_every == 0:
            bg_frames.append(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
        # keep only the two signal-head neighbourhoods (generous 80 px margin for alignment drift)
        crops = {}
        for k, g in scene0.signals.items():
            x, y = g["red"]
            x0, y0 = max(0, x - 80), max(0, y - 80)
            crops[k] = (x0, y0, small[y0:y + 80, x0:x + 80].copy())
        lamp_frames.append((t, crops))
        if r.boxes is not None and r.boxes.id is not None:
            for b, i, c, p in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.id.cpu().numpy(),
                                  r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                rows.append([round(t, 3), int(i), int(c), round(float(p), 3),
                             round(float(b[0] * sx), 1), round(float(b[1] * sy), 1),
                             round(float(b[2] * sx), 1), round(float(b[3] * sy), 1)])
        n += 1
    t_track = time.perf_counter()
    log(f"  tracking: {n} samples, {len(rows)} boxes, {t_track - t_start:.1f}s")

    # alignment
    ref = cv2.imread(str(REF_BG), cv2.IMREAD_GRAYSCALE)
    bg = np.median(np.stack(bg_frames), axis=0).astype(np.uint8) if bg_frames else ref
    Hm, ninl = estimate_homography(ref, bg)
    scene = Scene(REF_W, REF_H, Hm)
    log(f"  alignment: {describe(Hm)} ({ninl} inliers)")

    # signal states from the stored lamp crops, using aligned lamp positions
    times, scores = [], []
    for t, crops in lamp_frames:
        s = []
        for k in ("signal_main", "signal_left"):
            g = scene.signals[k]
            x0, y0, crop = crops[k]
            rad = g["radius"]
            for which, kind in (("red", "r"), ("green", "g")):
                x, y = g[which][0] - x0, g[which][1] - y0
                p = crop[max(0, y - rad):y + rad + 1, max(0, x - rad):x + rad + 1].astype(np.float32)
                if p.size == 0:
                    s.append(0.0)
                    continue
                b, gg, rr = p[..., 0], p[..., 1], p[..., 2]
                s.append(float(np.mean(rr - (gg + b) / 2)) if kind == "r" else float(np.mean((gg + b) / 2 - rr)))
        times.append(t)
        scores.append(s)
    times = np.array(times)
    states = states_offline(np.array(scores, dtype=np.float32)) if len(scores) else []

    def signal_by_t(q: float) -> str:
        if not states:
            return "unknown"
        k = int(np.searchsorted(times, q))
        return states[min(max(k, 0), len(states) - 1)]

    # rules
    engine = RuleEngine(scene, fps, stride)
    events = engine.run(rows, signal_by_t)
    events = [[max(0.0, s), min(duration, e), l] for s, e, l in events if e > s]
    log(f"  rules: {len(events)} events, total {time.perf_counter() - t_start:.1f}s for {duration:.1f}s of video")
    if return_debug:
        return events, {"rows": rows, "H": Hm.tolist(), "signal_t": times.tolist(), "signal": states,
                        "fps": fps, "n_frames": n_frames, "stride": stride}
    return events
