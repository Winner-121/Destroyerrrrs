# Destroyerrrrs — WIUT Hackathon 2026, Computer Vision track

Traffic-event detection from a fixed road camera (Part A) and causal accident
anticipation (Part B), implemented behind the organizers' `solution.py` interface.

> Status: Part A complete and tuned against the team's labels of the four sample videos
> (`labels/`, `predictions_samples.json`). Part B is intentionally left as the default
> estimator (returns 0).

## Install and run

```bash
# Python 3.10+ (developed on 3.11)
pip install -r requirements.txt
# If torch does not see your GPU, install the CUDA build first, e.g.
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
bash weights/download.sh          # once, with internet: fetches yolo11m.pt (39 MB)

python run_submission.py --videos /data/test --out predictions.json --team Destroyerrrrs
python evaluate.py --pred predictions.json --validate-only
python evaluate.py --pred predictions.json --gt ground_truth.json
```

`run_submission.py` and `evaluate.py` are the unchanged starter-kit files.
No internet is needed at run time. The pipeline picks the GPU when available and
falls back to CPU.

Measured runtime on the samples (RTX 5060 laptop GPU, 8 cores): about 0.64x video
duration for Part A plus the harness's own Part B decode, against a 3x budget.

## Approach

```
video (4K, 30 fps)
  └─ every 3rd frame ─► YOLO11m (COCO, 1280 px) ─► ByteTrack ─► trajectories
  └─ 25 sampled frames ─► median background ─► SIFT homography to the reference layout
  └─ lamp pixels of two signal heads ─► red / green / amber per sample
trajectories + aligned layout + signal ─► rule engine ─► [start, end, label] segments
```

**Learned:** the object detector (YOLO11m, COCO-pretrained, no fine-tuning) and the
appearance-free tracker (ByteTrack). **Rule-based:** everything else.

* `src/scene_layout.json` — hand-drawn layout on `src/reference_bg.jpg` (1280x720):
  carriageway, two zebra crossings plus the side-street crossing, islands, stop
  line, near-carriageway approach, intersection box, parked-car zone, lamp
  positions of the two signal heads.
* `src/align.py` — the camera drifts between recordings (~1-2 % zoom, ~1° roll,
  up to 40 px). A homography from the reference background to each video's
  median background warps the layout; identity fallback if the match is weak.
* `src/signal.py` — signal state from lamp-pixel colour scores of both heads
  (same phase, both control the near carriageway); per-video adaptive thresholds
  offline, fixed thresholds with hysteresis online.
* `src/events.py` — per-class rules on trajectories, then merge fragments (gap
  ≤ 1 s), drop blips, union overlapping same-class segments.
* `src/flow_field.npz` — mean vehicle heading per 40 px cell learned from the
  sample trajectories; used for `wrong_way`.

### Classes predicted

| class | rule (short) |
|---|---|
| `red_light` | came down the near carriageway with the flow, crosses the stop line moving, signal red for ≥ 6 s before and ≥ 3 s after (the near-pole head has no amber: it goes red 6 s before the far-pole head), clears the zebra within 4 s |
| `stop_line` | stationary with the front 8–60 px past the stop bar while red, ≥ 3 s |
| `stopped_vehicle` | stationary ≥ 10 s inside the intersection or ≥ 60 s elsewhere; signal queues (upstream of the stop line while not green, or < 20 s after green) excluded; re-identified parked vehicles joined across track ids |
| `jaywalking` | pedestrian ≥ 14 px inside the carriageway, off any crossing/island/parked zone, walking ≥ 40 px, ≥ 2 s; riders inside vehicle boxes ignored |
| `failure_to_yield` | vehicle crossing the main zebra (with the flow, not along it) while a walking pedestrian is ahead of its front and inside, or about to enter, its lane corridor |
| `wrong_way` | heading opposite the learned flow (cos < −0.9) for ≥ 2 s outside the intersection |
| `solid_line_crossing` | ground point drifts monotonically from one side of a solid divider to the other; segment from the start of the drift to 1.5 s after settling |
| `congestion` | ≥ 4 vehicles stationary inside the intersection box at once for ≥ 5 s, extended while ≥ 2 remain |

Not predicted (rules tried and dropped after review against the team labels):
`illegal_turn` (island-mounting test, 0/11 confirmed), `illegal_u_turn` (heading reversal,
fires on tracker noise), `accident`, `near_miss`, `road_obstacle`, `fire_smoke`.
`CLASSES` in `solution.py` lists only the eight predicted ids.

### Results on the sample videos (team labels)

Labels: 4 videos, 18 min, 102 events by the team (`labels/my_labels.json`), scored with
the official `evaluate.py` (same-class overlapping labels merged first, as the
organizers do for simultaneous events). Boundaries in the labels are approximate.

| class | labels | F1@0.3 | F1@0.5 | F1@0.7 |
|---|---|---|---|---|
| stopped_vehicle | 6 | 0.83 | 0.83 | 0.83 |
| stop_line | 4 | 0.73 | 0.73 | 0.73 |
| red_light | 1 | 1.00 | 1.00 | 1.00 |
| congestion | 2 | 1.00 | 1.00 | 0.00 |
| jaywalking | 33 | 0.63 | 0.49 | 0.44 |
| solid_line_crossing | 22 | 0.60 | 0.54 | 0.38 |
| failure_to_yield | 11 | 0.05 | 0.05 | 0.05 |
| wrong_way / illegal_turn / illegal_u_turn | 1 each | 0 | 0 | 0 |

Score A on the team labels: **0.43** (macro over the 10 classes present in labels or predictions).

### Data and models

| item | source | licence |
|---|---|---|
| YOLO11m weights | Ultralytics assets (COCO-pretrained) | AGPL-3.0 |
| ByteTrack | Ultralytics implementation | AGPL-3.0 |
| scene layout, flow field | drawn/derived from the 4 sample videos | ours |

No external video datasets were used so far.

### Determinism

Seeds fixed (`src/pipeline.set_seeds`, seed 0), cuDNN benchmark off. Inference is
FP16 on GPU; repeated runs on the same machine give identical events.

## Repository layout

```
solution.py              interface: CLASSES, detect_events, RiskEstimator
run_submission.py        organizers' harness (unchanged)
evaluate.py              organizers' metric (unchanged)
requirements.txt
weights/download.sh      fetches yolo11m.pt
src/                     pipeline, alignment, scene, signal, rules, layout
tools/                   dev tools: labelling web app, proxy-based rule runner, event reviewer
labels/                  team labels of the sample videos (ground-truth format) + rough notes
predictions_samples.json output of the harness on the 4 sample videos
notebooks/               EDA (to come)
```

### Dev tools

* `tools/label_server.py` — local web app to label the sample videos frame by frame;
  writes `labels/my_labels.json` in the evaluator's ground-truth format.
* `tools/run_rules_proxy.py` — runs alignment + signal + rules on 720p proxies with cached tracks.
* `tools/review_event.py` — renders a frame strip with boxes for one candidate event.

## Team

| member | role | links |
|---|---|---|
| TBD | TBD | TBD |
| TBD | TBD | TBD |
| TBD | TBD | TBD |

## What worked / what did not (running notes)

* Reading the traffic signal directly from lamp pixels works in daylight and at dusk
  once the green lamp is scored as cyan; the near-pole head is the reliable one in sun.
* Early-red running is common at this intersection; every clip has cars crossing 2-5 s into red.
* Per-video alignment was necessary: the "fixed" camera moves between recordings.
* Open questions awaiting labels: how annotators treat cars parked in the far kerb
  lane at the bus stop, and short (< 1 s) failure-to-yield passes.
