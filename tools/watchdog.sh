#!/usr/bin/env bash
# Keeps the labelling server and the public tunnel alive. Run: setsid -f tools/watchdog.sh
cd "$(dirname "$0")/.."
while true; do
  pgrep -f "[l]abel_server.py" >/dev/null || { setsid -f .venv/bin/python tools/label_server.py 8765 >> work/label_server.log 2>&1 < /dev/null; echo "$(date) restarted server" >> work/watchdog.log; }
  pgrep -f "[c]loudflared tunnel" >/dev/null || { setsid -f /data/bin/cloudflared tunnel --url http://localhost:8765 --no-autoupdate > work/tunnel.log 2>&1 < /dev/null; echo "$(date) restarted tunnel" >> work/watchdog.log; }
  sleep 30
done
