"""Gradio demo for Hugging Face Spaces (CPU): upload an .mp4 from the camera, get events + timeline + annotated preview."""
import json, os, sys, time, tempfile, subprocess
from pathlib import Path
import cv2, numpy as np, gradio as gr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
from src.pipeline import detect_events as _detect

MAX_SEC = 125
COL = {"red_light": (255, 77, 77), "stop_line": (255, 159, 67), "stopped_vehicle": (255, 212, 59), "jaywalking": (76, 201, 240),
       "failure_to_yield": (243, 104, 224), "wrong_way": (177, 151, 252), "solid_line_crossing": (81, 207, 102), "congestion": (255, 135, 135)}

def probe(path):
    cap = cv2.VideoCapture(path); fps = cap.get(cv2.CAP_PROP_FPS) or 25; n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(3)), int(cap.get(4)); cap.release(); return fps, n, w, h

def timeline_png(events, dur):
    W, H = 900, 40 + 22 * max(1, len(COL)); img = np.full((H, W, 3), 24, np.uint8)
    rows = list(COL)
    for i, c in enumerate(rows):
        y = 30 + 22 * i; cv2.putText(img, c, (4, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
        for s, e, l in events:
            if l != c: continue
            x0, x1 = int(150 + s / dur * (W - 160)), int(150 + e / dur * (W - 160))
            cv2.rectangle(img, (x0, y), (max(x1, x0 + 2), y + 16), COL[c][::-1], -1)
    for m in range(0, int(dur) + 1, 30):
        x = int(150 + m / dur * (W - 160)); cv2.line(img, (x, 22), (x, H - 4), (60, 60, 60), 1); cv2.putText(img, f"{m//60}:{m%60:02d}", (x - 12, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1)
    return img[:, :, ::-1]

def annotate(path, events, dur, out_path, max_frames=1800):
    cap = cv2.VideoCapture(path); fps = cap.get(cv2.CAP_PROP_FPS) or 25; n = int(cap.get(7))
    step = max(1, int(round(n / max_frames))); W, H = 960, 540
    tmp = out_path + ".raw.mp4"; out = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), max(5, fps / step), (W, H + 40)); idx = 0
    while True:
        ok = cap.grab()
        if not ok: break
        if idx % step == 0:
            ok, f = cap.retrieve(); t = idx / fps; f = cv2.resize(f, (W, H)); y = 24
            for s, e, l in events:
                if s <= t <= e:
                    cv2.putText(f, f"{l} {s:.1f}-{e:.1f}s", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COL[l][::-1], 2); y += 24
            cv2.putText(f, f"{int(t//60)}:{t%60:04.1f}", (W - 90, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            bar = np.full((40, W, 3), 25, np.uint8)
            for s, e, l in events:
                cv2.rectangle(bar, (int(s / dur * W), 8), (max(int(s / dur * W) + 2, int(e / dur * W)), 32), COL[l][::-1], -1)
            cx = int(t / dur * W); cv2.line(bar, (cx, 2), (cx, 38), (255, 255, 255), 2)
            out.write(np.vstack([f, bar]))
        idx += 1
    out.release(); cap.release()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-c:v", "libx264", "-preset", "veryfast", "-crf", "30", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path], check=False)
    if os.path.exists(tmp): os.remove(tmp)
    return out_path if os.path.exists(out_path) else None

def run(video, progress=gr.Progress()):
    if video is None: return "Upload an .mp4 first.", None, None, None
    fps, n, w, h = probe(video); dur = n / fps
    if dur > MAX_SEC: return f"Video is {dur:.0f} s; the demo accepts up to {MAX_SEC} s.", None, None, None
    progress(0.05, desc="Detecting and tracking (CPU, this takes a few minutes)…")
    t0 = time.perf_counter(); log = []
    events = _detect(video, stride=5, imgsz=960, log=lambda m: log.append(m))
    progress(0.75, desc="Rendering preview…")
    out = os.path.join(tempfile.mkdtemp(), "annotated.mp4"); vid = annotate(video, events, dur, out)
    progress(1.0)
    rows = [[f"{s//60:.0f}:{s%60:04.1f}", f"{e//60:.0f}:{e%60:04.1f}", l, f"{e-s:.1f}"] for s, e, l in events]
    txt = f"{len(events)} events in {dur:.0f} s of video · processed in {time.perf_counter()-t0:.0f} s\n" + "\n".join(log)
    return txt, rows, timeline_png(events, dur), vid

with gr.Blocks(title="Destroyerrrrs — traffic event detection") as demo:
    gr.Markdown("## Destroyerrrrs — traffic events from the fixed camera\nUpload an .mp4 from the same camera (≤ 2 min). CPU inference with a lighter setting (every 5th frame at 960 px); expect a few minutes for a 2-minute clip. [Repository](https://github.com/Winner-121/Destroyerrrrs) · [Team site](https://winner-121.github.io/Destroyerrrrs/)")
    with gr.Row():
        inp = gr.Video(label="Input video (.mp4, ≤ 2 min)", sources=["upload"])
        with gr.Column():
            btn = gr.Button("Detect events", variant="primary"); status = gr.Textbox(label="Status", lines=4)
    tl = gr.Image(label="Timeline", type="numpy")
    with gr.Row():
        table = gr.Dataframe(headers=["start", "end", "class", "duration s"], label="Events")
        vid = gr.Video(label="Annotated preview")
    btn.click(run, inputs=inp, outputs=[status, table, tl, vid])
demo.queue(max_size=8).launch()
