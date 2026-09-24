"""Render a frame strip for an event window with tracked boxes, to eyeball a candidate.
usage: python tools/review_event.py VIDEO START END OUT [--every SEC] [--hl ID,ID]"""
import sys, json, argparse, cv2, numpy as np
sys.path.insert(0, "src")
ap = argparse.ArgumentParser(); ap.add_argument("video"); ap.add_argument("start", type=float); ap.add_argument("end", type=float); ap.add_argument("out")
ap.add_argument("--every", type=float, default=1.0); ap.add_argument("--hl", default=""); ap.add_argument("--crop", default="0,150,1280,720"); ap.add_argument("--pad", type=float, default=2.0)
a = ap.parse_args()
names = {0: "person", 1: "bike", 2: "car", 3: "moto", 5: "bus", 7: "truck"}
d = json.load(open(f"work/tracks/{a.video}_proxy.json")); fps = d["fps"]
by_t = {}
for r in d["rows"]: by_t.setdefault(round(r[0], 2), []).append(r)
ts = np.array(sorted(by_t)); hl = {int(x) for x in a.hl.split(",") if x}
x0, y0, x1, y1 = map(int, a.crop.split(","))
cap = cv2.VideoCapture(f"work/proxy/{a.video}.mp4"); tiles = []
t = a.start - a.pad
while t <= a.end + a.pad + 1e-6:
    cap.set(1, int(round(t * fps))); ok, f = cap.read()
    if not ok: break
    k = ts[min(len(ts) - 1, int(np.searchsorted(ts, t)))]
    for _, i, c, cf, bx1, by1, bx2, by2 in by_t.get(k, []):
        col = (0, 0, 255) if int(i) in hl else ((0, 255, 0) if c == 0 else (255, 200, 0))
        th = 3 if int(i) in hl else 1
        cv2.rectangle(f, (int(bx1), int(by1)), (int(bx2), int(by2)), col, th)
        if int(i) in hl or c != 0: cv2.putText(f, f"{names.get(int(c), c)}#{int(i)}", (int(bx1), int(by1) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    f = f[y0:y1, x0:x1]; f = cv2.resize(f, (640, int(640 * f.shape[0] / f.shape[1])))
    tag = "IN" if a.start <= t <= a.end else "  "
    cv2.putText(f, f"{int(t//60)}:{t%60:04.1f} {tag}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    tiles.append(f); t += a.every
cols = 3
rows = [np.hstack(tiles[i:i + cols] + [np.zeros_like(tiles[0])] * (cols - len(tiles[i:i + cols]))) for i in range(0, len(tiles), cols)]
cv2.imwrite(a.out, np.vstack(rows)); print("wrote", a.out, len(tiles), "frames")
