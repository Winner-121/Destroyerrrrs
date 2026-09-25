"""Score proxy-rule events against the team labels, with same-class overlapping labels
merged into one segment (the organizers' convention for simultaneous events)."""
import json, sys, subprocess
sys.path.insert(0, "src")
from events import merge_intervals
gt = json.load(open("labels/my_labels.json"))
for v in gt:
    by = {}
    for s, e, l in gt[v]["events"]:
        by.setdefault(l, []).append((s, e))
    gt[v]["events"] = sorted([[s, e, l] for l, segs in by.items() for s, e in merge_intervals(segs)])
json.dump(gt, open("work/gt_merged.json", "w"), indent=1)
ev = json.load(open("work/rules_proxy_events.json"))
pred = {"team": "dev", "videos": {f"{k}.MP4": {"events": v, "risk": []} for k, v in ev.items()}}
json.dump(pred, open("work/pred_proxy_all.json", "w"))
out = subprocess.run([sys.executable, "evaluate.py", "--pred", "work/pred_proxy_all.json", "--gt", "work/gt_merged.json"], capture_output=True, text=True).stdout
print("\n".join(l for l in out.splitlines() if l.startswith("class") or l.startswith("Score A") or (l[:1].islower() and l.split()[0].replace("_", "").isalpha() and len(l.split()) >= 5)))
