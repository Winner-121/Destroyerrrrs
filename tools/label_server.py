#!/usr/bin/env python3
"""Tiny local labelling server.

    python tools/label_server.py            # then open http://localhost:8765

Serves the 720p proxy videos (with HTTP Range so seeking works), the labelling
page, and reads/writes labels/<video>.json. After every save it regenerates
labels/my_labels.json in the evaluate.py ground-truth format using fps/duration
from the ORIGINAL videos in Samples/.
"""
from __future__ import annotations

import json
import os
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
PROXY = ROOT / "work" / "proxy"
SAMPLES = ROOT / "Samples"
LABELS = ROOT / "labels"
LABELS.mkdir(exist_ok=True)
PAGE = Path(__file__).with_name("label_page.html")


def video_meta() -> dict:
    meta = {}
    for p in sorted(SAMPLES.glob("*.MP4")) + sorted(SAMPLES.glob("*.mp4")):
        cap = cv2.VideoCapture(str(p))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        meta[p.name] = {"fps": float(fps), "n_frames": n, "duration": n / fps, "proxy": p.stem + ".mp4"}
    return meta


META = video_meta()


def write_ground_truth() -> None:
    gt = {}
    for name, m in META.items():
        f = LABELS / (Path(name).stem + ".json")
        events = []
        if f.exists():
            events = [[e["start"], e["end"], e["label"]] for e in json.loads(f.read_text()).get("events", [])]
        gt[name] = {"duration": round(m["duration"], 3), "fps": round(m["fps"], 3), "events": sorted(events)}
    (LABELS / "my_labels.json").write_text(json.dumps(gt, indent=1))


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code, body: bytes, ctype="application/json", extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        if self.path == "/meta":
            return self._send(200, json.dumps(META).encode())
        m = re.match(r"^/candidates/([A-Za-z0-9_\-]+)$", self.path)
        if m:
            c = LABELS / "candidates.json"
            data = json.loads(c.read_text()).get(m.group(1), []) if c.exists() else []
            return self._send(200, json.dumps(data).encode())
        m = re.match(r"^/labels/([A-Za-z0-9_\-]+)$", self.path)
        if m:
            f = LABELS / (m.group(1) + ".json")
            notes = LABELS / "notes.json"
            data = json.loads(f.read_text()) if f.exists() else {"events": [], "rev": 0}
            data.setdefault("rev", 0)
            data.setdefault("rejected", [])
            data["notes"] = json.loads(notes.read_text()).get(m.group(1), []) if notes.exists() else []
            return self._send(200, json.dumps(data).encode())
        m = re.match(r"^/video/([A-Za-z0-9_\-]+\.mp4)$", self.path)
        if m:
            return self._range_file(PROXY / m.group(1))
        self._send(404, b"not found", "text/plain")

    def _range_file(self, p: Path):
        if not p.exists():
            return self._send(404, b"no video", "text/plain")
        size = p.stat().st_size
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)", rng)
            if m:
                if m.group(1):
                    start = int(m.group(1))
                if m.group(2):
                    end = min(int(m.group(2)), size - 1)
        length = end - start + 1
        self.send_response(206 if rng else 200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if rng:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with open(p, "rb") as fh:
            fh.seek(start)
            remaining = length
            while remaining > 0:
                chunk = fh.read(min(1 << 20, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                remaining -= len(chunk)

    def do_POST(self):
        m = re.match(r"^/labels/([A-Za-z0-9_\-]+)$", self.path)
        if not m:
            return self._send(404, b"not found", "text/plain")
        n = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(n) or b"{}")
        f = LABELS / (m.group(1) + ".json")
        cur_rev = json.loads(f.read_text()).get("rev", 0) if f.exists() else 0
        if int(body.get("rev", -1)) != cur_rev:  # someone else saved since this client loaded
            return self._send(409, json.dumps({"ok": False, "rev": cur_rev, "error": "stale"}).encode())
        events = []
        for e in body.get("events", []):
            s, t, lab = float(e["start"]), float(e["end"]), str(e["label"])
            if 0 <= s < t:
                events.append({"start": round(s, 2), "end": round(t, 2), "label": lab, "note": e.get("note", "")})
        events.sort(key=lambda e: (e["start"], e["label"]))
        rejected = []
        for r in body.get("rejected", []):
            try:
                rejected.append({"start": round(float(r["start"]), 2), "end": round(float(r["end"]), 2), "label": str(r["label"])})
            except Exception:
                pass
        new_rev = cur_rev + 1
        f.write_text(json.dumps({"events": events, "rejected": rejected, "rev": new_rev, "by": body.get("who", "")}, indent=1))
        write_ground_truth()
        self._send(200, json.dumps({"ok": True, "n": len(events), "rev": new_rev}).encode())


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    write_ground_truth()
    print(f"labelling tool: http://localhost:{port}   videos: {', '.join(META)}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
