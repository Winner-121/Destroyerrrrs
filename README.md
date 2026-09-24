# Destroyerrrrs — WIUT Hackathon 2026, Computer Vision track

Traffic-event detection from a fixed road camera (Part A) and causal accident
anticipation (Part B), implemented behind the organizers' `solution.py` interface.

> Status: Part A pipeline complete and validated with the official harness on the
> four sample videos (`predictions_samples.json`). Part B is a placeholder for now.
> Team labels of the sample videos are in progress (`labels/`).

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
| `red_light` | front crosses the stop line ≥ 1.5 s into red, moving, and clears the zebra within 4 s |
| `stop_line` | stationary with the front past the stop bar (not through the zebra) while red |
| `stopped_vehicle` | stationary ≥ 10 s inside the intersection, or ≥ 60 s elsewhere; signal queues and whole-clip parked cars excluded |
| `jaywalking` | pedestrian ≥ 14 px inside the carriageway, off any crossing/island, walking ≥ 40 px, ≥ 2 s; riders inside vehicle boxes ignored |
| `failure_to_yield` | vehicle moving across the main zebra within 220 px of a pedestrian on it |
| `wrong_way` | heading opposite the learned flow (cos < −0.8) for ≥ 3 s outside the intersection |

`CLASSES` in `solution.py` lists only these six; other official ids are never predicted.

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
