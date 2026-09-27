"""Deploy the Gradio demo on Modal (free tier, 8 CPU cores, scales to zero).

    modal deploy demo/modal_app.py

The container image bundles src/, weights/ and demo/app.py from this repository.
"""
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        "numpy==2.4.6", "opencv-python-headless==5.0.0.93", "scipy>=1.10", "lap>=0.4",
        "ultralytics==8.4.161", "torch>=2.3", "torchvision>=0.18", "gradio==6.28.0",
    )
    .add_local_dir(ROOT / "src", remote_path="/root/app/src")
    .add_local_dir(ROOT / "weights", remote_path="/root/app/weights")
    .add_local_file(ROOT / "demo" / "app.py", remote_path="/root/app/app.py")
)

app = modal.App("destroyerrrrs-traffic-events", image=image)


@app.function(
    cpu=8.0,
    memory=16384,
    timeout=1800,
    scaledown_window=300,
    max_containers=2,
)
@modal.concurrent(max_inputs=4)
@modal.asgi_app()
def web():
    import os
    import sys

    os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
    os.environ["DEMO_STRIDE"] = "6"
    os.environ["DEMO_IMGSZ"] = "960"
    os.environ["DEMO_MAX_SEC"] = "125"
    sys.path.insert(0, "/root/app")
    os.chdir("/root/app")
    import gradio as gr
    from fastapi import FastAPI

    import app as demo_module  # noqa: F401  (defines `demo`, does not launch: guarded by DEMO_NO_LAUNCH)

    api = FastAPI()
    return gr.mount_gradio_app(api, demo_module.demo.queue(max_size=8), path="/", max_file_size="2gb")
