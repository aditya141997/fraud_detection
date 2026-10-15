#!/usr/bin/env bash
# Latency at fixed request rates against the running stack (API on :8000, Redis, Redpanda, Prometheus,
# Grafana and the drift monitor all up). Each step: 70 s including a short ramp; users x 4 req/s = rate.
# Prerequisite: python -m src.serving.replay_score --limit 3000   (puts the events' features in Redis)
cd "$(dirname "$0")/.."
mkdir -p ${OUT:-loadtest/results/final}
export PYTHONPATH=. N_EVENTS=3000 RATE_PER_USER=4 MLFLOW_DISABLE_AGENT_HINT=1
for users in 25 50 100 150; do
  .venv/Scripts/python -W ignore -m locust -f loadtest/locustfile.py ScoreUser --headless -u $users -r 25 \
      -t 70s --host http://127.0.0.1:8000 --csv ${OUT:-loadtest/results/final}/rps$((users*4)) --only-summary \
      > ${OUT:-loadtest/results/final}/rps$((users*4)).log 2>&1
done
echo done > ${OUT:-loadtest/results/final}/DONE
