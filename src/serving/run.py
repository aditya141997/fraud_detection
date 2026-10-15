"""Launch the scoring API with several worker processes.

One Python process tops out around one core of GIL-bound work (request parsing, Feast lookup, JSON), so
latency under load needs more than one worker. Prometheus metrics are aggregated across workers through
files in PROMETHEUS_MULTIPROC_DIR, which is wiped on every start.

    python -m src.serving.run --workers 4 --port 8000
"""
import argparse
import os
import shutil
import tempfile
from pathlib import Path

import uvicorn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    if a.workers > 1:
        d = Path(tempfile.gettempdir()) / "fraud_prom_multiproc"
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = str(d)
        os.environ["FRAUD_PROM_MULTIPROC"] = "1"
    uvicorn.run("src.serving.app:app", host="0.0.0.0", port=a.port, workers=a.workers,
                log_level="warning")


if __name__ == "__main__":
    main()
