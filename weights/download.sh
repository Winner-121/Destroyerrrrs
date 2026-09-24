#!/usr/bin/env bash
# Fetches the detector weights (run once with internet, before the offline evaluation).
# YOLO11m, COCO-pretrained, AGPL-3.0, from the Ultralytics assets release.
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -f yolo11m.pt ]; then
  curl -L -o yolo11m.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11m.pt
fi
ls -la yolo11m.pt
