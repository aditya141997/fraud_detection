# Deploying the scoring API on Railway

CI (`.github/workflows/ci.yml`): pytest, then build the Docker image and check `/health` against a Redis container.
CD: Railway's own GitHub integration deploys every push to `main` (enable "Wait for CI" on the service so a red CI blocks the deploy).

## One-time Railway setup
1. New project; add a **Redis** plugin.
2. Add a service named `api` from this GitHub repo. `railway.toml` selects the Dockerfile and `/health` check.
3. On `api`, set variable `REDIS_URL` = the Redis plugin's `${{Redis.REDIS_URL}}`. Optional: `WORKERS` (default 2),
   `PUBLISH_SCORED_EVENTS=0` until a Kafka broker exists, otherwise `KAFKA_BOOTSTRAP`.
4. Generate a public domain for `api`.
5. Settings: turn on "Wait for CI". Set `PORT` to the domain's target port, `WORKERS=1`, `EXPLAIN_EAGER=0` on small plans.

## What the image contains
`deploy/model/model_v1.*` is registered as `champion` in a container-local MLflow registry on first start
(`src/serving/seed.py`). `deploy/feast/` holds the Feast registry and a `feature_store.yaml` that takes Redis from
`REDIS_CONNECTION_STRING` (built from `REDIS_URL` by `deploy/entrypoint.sh`).

## Not covered (yet)
- **Redis starts empty.** `/score` returns `missing_entities` until something writes velocity features. That needs
  the Kafka consumer (`src.ingestion.consumer`) plus a Redpanda broker, which are not deployed here.
- **Promotion does not persist.** The registry lives inside the container, so a redeploy resets the champion
  to `model_v1`. Use a shared MLflow server (Postgres + artifact storage) to keep alias moves.
- **Drift monitor and retraining** need `data/processed/*.parquet` (not in git) and are not deployed.
- Latency on Railway has not been measured; the README numbers are for a single local machine.
- The Docker image was not built locally (Docker Desktop was not running); the CI `docker` job is its first build.
