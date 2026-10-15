"""Locust load test for the scoring API.

Prerequisite: the replay events must already have their online features in Redis, e.g.
    python -m src.serving.replay_score --limit 2000

Fixed request rate (each user sends RATE_PER_USER requests/second), so the latency percentiles are
measured at a known load instead of whatever the client can push:

    locust -f loadtest/locustfile.py ScoreUser --headless -u 50 -r 25 -t 90s --host http://localhost:8000 \
        --csv loadtest/results/rps200 --only-summary
    # 50 users x 4 req/s = 200 req/s (RATE_PER_USER env var changes the per-user rate)
"""
import os
import random

import pandas as pd
from locust import FastHttpUser, constant_throughput, task

from src.data import PROCESSED
from src.serving.replay_score import payload

N_EVENTS = int(os.environ.get("N_EVENTS", 2000))
RATE = float(os.environ.get("RATE_PER_USER", 4))

# loaded once per worker process; every payload is a real replay transaction
EVENTS = [payload(r) for r in pd.read_parquet(PROCESSED / "replay.parquet").head(N_EVENTS).to_dict("records")]


class ScoreUser(FastHttpUser):
    wait_time = constant_throughput(RATE)

    @task
    def score(self):
        with self.client.post("/score", json=random.choice(EVENTS), name="/score", catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"HTTP {r.status_code}")


class ExplainUser(FastHttpUser):
    """SHAP reasons for flagged transactions only (~0.4 s each); run separately, not on the scoring path."""
    wait_time = constant_throughput(0.5)

    @task
    def explain(self):
        self.client.post("/explain", json=random.choice(EVENTS), name="/explain")
