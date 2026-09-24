"""Per-video alignment of the fixed-camera layout.

The camera is 'fixed' but between recording sessions it drifts by a few
pixels, ~1 degree of roll and ~1-2 % of zoom. We estimate a homography from
the reference background (on which the layout was drawn) to the current
video's background (median of sampled frames) with SIFT + RANSAC, and apply
it to every polygon. Falls back to identity when the match is weak.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REF_W, REF_H = 1280, 720
MIN_INLIERS = 40


def median_background(video_path: str, n_frames: int = 25, width: int = REF_W) -> np.ndarray:
    """Grayscale median of n frames spread over the video, resized to reference width."""
    cap = cv2.VideoCapture(video_path)
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for k in np.linspace(0, max(0, N - 1), n_frames).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
        ok, f = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        if g.shape[1] != width:
            g = cv2.resize(g, (width, int(round(g.shape[0] * width / g.shape[1]))), interpolation=cv2.INTER_AREA)
        frames.append(g)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames read from {video_path}")
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def estimate_homography(ref_bg: np.ndarray, bg: np.ndarray) -> tuple[np.ndarray, int]:
    """H maps reference-pixel coords -> this video's coords (both at reference size)."""
    sift = cv2.SIFT_create(4000)
    kr, dr = sift.detectAndCompute(ref_bg, None)
    kb, db = sift.detectAndCompute(bg, None)
    if dr is None or db is None or len(kr) < 10 or len(kb) < 10:
        return np.eye(3), 0
    matches = cv2.BFMatcher().knnMatch(dr, db, k=2)
    good = [m for m, n in (p for p in matches if len(p) == 2) if m.distance < 0.7 * n.distance]
    if len(good) < MIN_INLIERS:
        return np.eye(3), len(good)
    src = np.float32([kr[m.queryIdx].pt for m in good])
    dst = np.float32([kb[m.trainIdx].pt for m in good])
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    n_inl = int(inl.sum()) if inl is not None else 0
    if H is None or n_inl < MIN_INLIERS:
        return np.eye(3), n_inl
    # sanity: reject wild transforms (the camera only drifts a little)
    A = H[:2, :2]
    scale = float(np.sqrt(abs(np.linalg.det(A))))
    c = H @ np.array([REF_W / 2, REF_H / 2, 1.0])
    shift = np.hypot(c[0] / c[2] - REF_W / 2, c[1] / c[2] - REF_H / 2)
    if not (0.9 < scale < 1.1) or shift > 120:
        return np.eye(3), n_inl
    return H, n_inl


def align_video(video_path: str, ref_path: Path = HERE / "reference_bg.jpg") -> tuple[np.ndarray, int]:
    ref = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE)
    bg = median_background(video_path)
    return estimate_homography(ref, bg)


def describe(H: np.ndarray) -> str:
    A = H[:2, :2]
    scale = float(np.sqrt(abs(np.linalg.det(A))))
    rot = float(np.degrees(np.arctan2(A[1, 0], A[0, 0])))
    c = H @ np.array([REF_W / 2, REF_H / 2, 1.0])
    return f"shift ({c[0] / c[2] - REF_W / 2:+.1f},{c[1] / c[2] - REF_H / 2:+.1f}) px, scale {scale:.4f}, rot {rot:+.2f} deg"
