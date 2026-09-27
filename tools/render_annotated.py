"""Render an annotated 640p version of a proxy video: tracked boxes, event banner, timeline bar.
usage: python tools/render_annotated.py C3905"""
import sys, json, cv2, numpy as np
v = sys.argv[1]
COL = {"red_light": (0, 0, 255), "stop_line": (0, 140, 255), "stopped_vehicle": (0, 200, 255), "jaywalking": (255, 200, 0),
       "failure_to_yield": (255, 0, 200), "wrong_way": (200, 0, 255), "solid_line_crossing": (0, 255, 120), "congestion": (255, 120, 0)}
names = {0: "person", 1: "bike", 2: "car", 3: "moto", 5: "bus", 7: "truck"}
d = json.load(open(f"work/tracks/{v}_proxy.json")); fps = d["fps"]
by_t = {}
for r in d["rows"]: by_t.setdefault(round(r[0], 2), []).append(r)
ts = np.array(sorted(by_t))
events = json.load(open("predictions_samples.json"))["videos"][f"{v}.MP4"]["events"]
cap = cv2.VideoCapture(f"work/proxy/{v}.mp4"); N = int(cap.get(7)); dur = N / fps
W, H = 960, 540
out = cv2.VideoWriter(f"work/site_videos/{v}_annot_raw.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 15, (W, H + 60))
idx = 0
while True:
    ok, f = cap.read()
    if not ok: break
    if idx % 2 == 0:
        t = idx / fps
        k = ts[min(len(ts) - 1, int(np.searchsorted(ts, t)))]
        sx, sy = W / f.shape[1], H / f.shape[0]
        f = cv2.resize(f, (W, H))
        for _, i, c, cf, x1, y1, x2, y2 in by_t.get(k, []):
            col = (80, 220, 80) if c == 0 else (255, 200, 80)
            cv2.rectangle(f, (int(x1 * sx), int(y1 * sy)), (int(x2 * sx), int(y2 * sy)), col, 1)
        active = [e for e in events if e[0] <= t <= e[1]]
        y = 26
        for s, e, l in active:
            cv2.rectangle(f, (8, y - 18), (8 + 9 * len(l) + 90, y + 6), (0, 0, 0), -1)
            cv2.putText(f, f"{l}  {s:.1f}-{e:.1f}s", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL.get(l, (255, 255, 255)), 2); y += 26
        cv2.putText(f, f"{int(t // 60)}:{t % 60:04.1f}", (W - 90, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        bar = np.zeros((60, W, 3), np.uint8) + 25
        for s, e, l in events:
            x0, x1 = int(s / dur * W), max(int(s / dur * W) + 2, int(e / dur * W))
            cv2.rectangle(bar, (x0, 12), (x1, 44), COL.get(l, (200, 200, 200)), -1)
        cx = int(t / dur * W); cv2.line(bar, (cx, 4), (cx, 56), (255, 255, 255), 2)
        out.write(np.vstack([f, bar]))
    idx += 1
out.release(); cap.release(); print(v, "raw done")
