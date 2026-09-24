"""Dev harness: align + signal + rules on the 720p proxies using cached tracks."""
import sys, json, time, cv2, numpy as np
from pathlib import Path
sys.path.insert(0, "src")
from align import align_video, describe
from scene import Scene
from signal import lamp_scores, states_offline
from events import RuleEngine

def fmt(t): return f"{int(t//60)}:{t%60:05.2f}"

def signal_scores(video, scene, stride, cache):
    if cache.exists():
        z = np.load(cache); return z["t"], z["s"]
    cap = cv2.VideoCapture(video); fps = cap.get(5); ts, ss, idx = [], [], 0
    while True:
        if not cap.grab(): break
        if idx % stride == 0:
            ok, f = cap.retrieve(); ts.append(idx / fps); ss.append(lamp_scores(f, scene))
        idx += 1
    cap.release(); t, s = np.array(ts), np.array(ss); np.savez(cache, t=t, s=s); return t, s

out = {}
for n in sys.argv[1:] or ["C3905", "C3902", "C3897", "C3896"]:
    video = f"work/proxy/{n}.mp4"; d = json.load(open(f"work/tracks/{n}_proxy.json"))
    t0 = time.perf_counter(); H, ninl = align_video(video); print(f"\n== {n}: alignment {describe(H)} ({ninl} inliers, {time.perf_counter()-t0:.1f}s)")
    sc = Scene(1280, 720, H)
    t, s = signal_scores(video, sc, d["stride"], Path(f"work/signal/{n}_proxy.npz"))
    st = states_offline(s)
    runs = []
    for tt, x in zip(t, st):
        if runs and runs[-1][0] == x: runs[-1][2] = tt
        else: runs.append([x, tt, tt])
    print("  signal: " + " | ".join(f"{a[:3]} {fmt(b)}-{fmt(c)}" for a, b, c in runs))
    idx = {round(float(tt), 2): x for tt, x in zip(t, st)}
    tarr = t
    def signal_by_t(q):
        k = int(np.searchsorted(tarr, q)); k = min(max(k, 0), len(st) - 1); return st[k]
    eng = RuleEngine(sc, d["fps"], d["stride"])
    ev = eng.run(d["rows"], signal_by_t)
    out[n] = ev
    print(f"  {len(ev)} events")
    for a, b, l in ev: print(f"    {fmt(a)} - {fmt(b)}  {l}  ({b-a:.1f}s)")
json.dump(out, open("work/rules_proxy_events.json", "w"), indent=1)
