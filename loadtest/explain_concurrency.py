"""/explain under concurrent clients (SHAP is ~0.4 s of CPU per call and runs in separate processes).

    python -m loadtest.explain_concurrency [url]
Prerequisite: the replayed events' features are in Redis (python -m src.serving.replay_score --limit 3000).
"""
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np
import pandas as pd

from src.data import PROCESSED
from src.serving.replay_score import payload

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
EVENTS = [payload(r) for r in pd.read_parquet(PROCESSED / "replay.parquet").head(120).to_dict("records")]


def client(i):
    lat, codes = [], []
    with httpx.Client(base_url=URL, timeout=30) as c:
        for e in EVENTS[i * 5:(i + 1) * 5]:
            s = time.perf_counter()
            try:
                codes.append(c.post("/explain", json=e).status_code)
            except httpx.HTTPError as x:
                codes.append(type(x).__name__)
            lat.append(time.perf_counter() - s)
    return lat, codes


for n in (1, 6, 10, 16):
    with ThreadPoolExecutor(n) as ex:
        res = list(ex.map(client, range(n)))
    lat = sum((r[0] for r in res), [])
    codes = Counter(str(c) for r in res for c in r[1])
    print(f"{n:>2} clients x5 requests: p50 {np.median(lat):.2f}s p95 {np.percentile(lat, 95):.2f}s "
          f"max {max(lat):.2f}s status {dict(codes)}")
