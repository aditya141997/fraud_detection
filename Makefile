PY=.venv/Scripts/python

data:
	$(PY) -m src.data

features:
	$(PY) -m src.features
	cd feature_repo && ../.venv/Scripts/feast apply

train:
	$(PY) -m src.training.train

test:
	$(PY) -m pytest -q

# ---- Week 3: serving, load test, monitoring
register:  # MLflow registry: model_v1 -> champion, model_v2 -> challenger
	$(PY) -m src.serving.registry

serve:  # scoring API, 4 worker processes, http://localhost:8000
	$(PY) -m src.serving.run --workers 4

replay-score:  # lock-step replay (writes online features, scores via the API, checks parity vs offline)
	$(PY) -m src.serving.replay_score --limit 3000

drift-baseline:  # learn per-column drift thresholds from healthy windows
	$(PY) -m src.monitoring.baseline 1000 20

drift:  # follow scored-events, Prometheus gauges on :8001, report in reports/drift_latest.html
	$(PY) -m src.monitoring.drift

loadtest:  # fixed-rate latency matrix (needs `make replay-score` first)
	bash loadtest/run_matrix.sh

# ---- Week 4: injected drift, retrain flow, shadow mode, promotion
# Scenario (src/pipelines/scenario.py): events [0, 8000) healthy, drifted from 8000, labels known for [0, 20000),
# challenger shadows the champion on [20000, 40000). Flush Redis first: docker compose exec redis redis-cli FLUSHALL
stream-phase1:  # live stream with injected drift, no shadow model yet (run `make serve` and `make drift` first)
	$(PY) -m src.pipelines.stream --start 0 --limit 20000

retrain:  # Prefect flow: fires only if the drift monitor's last 3 evaluations alerted (--force to override)
	$(PY) -m src.pipelines.retrain_flow retrain

serve-shadow:  # restart the API with the `challenger` alias scored in shadow next to the champion
	SHADOW_ALIAS=challenger $(PY) -m src.serving.run --workers 4

stream-phase2:  # shadow window: events [20000, 40000), drift continues
	$(PY) -m src.pipelines.stream --start 20000 --limit 40000

promote:  # Prefect flow: compare on labelled shadow events, apply the promotion rule, promote + reload if it passes
	$(PY) -m src.pipelines.retrain_flow promote
