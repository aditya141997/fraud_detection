"""FastAPI scoring service.

    uvicorn src.serving.app:app --port 8000

The model is whatever version holds the `champion` alias in the MLflow registry (`POST /reload` re-reads
the alias after a promotion). Velocity features come from Redis through Feast, so the Feature consumer
must already have processed a transaction before it is scored (see README, "Serving").
"""
import asyncio
import faulthandler
import logging
import os
import tempfile
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response
from prometheus_client import (CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Histogram, generate_latest,
                               multiprocess)
from pydantic import BaseModel, ConfigDict, Field

from src.serving import explain_worker
from src.serving.adapters import EventPublisher, FeastFetcher, NullPublisher
from src.serving.registry import alias_version, load_alias
from src.serving.scorer import Scorer, monitored_columns

log = logging.getLogger("fraud.serving")

ALIAS = os.environ.get("MODEL_ALIAS", "champion")
BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "localhost:19092")
PUBLISH = os.environ.get("PUBLISH_SCORED_EVENTS", "1") == "1"
SCORE_THREADS = int(os.environ.get("SCORE_THREADS", 8))
EXPLAIN_SLOTS = int(os.environ.get("EXPLAIN_SLOTS", 1))  # SHAP processes (and concurrent computations) per API worker
EXPLAIN_EAGER = os.environ.get("EXPLAIN_EAGER", "1") == "1"  # build the SHAP pool at startup
EXPLAIN_WAIT_S = float(os.environ.get("EXPLAIN_WAIT_S", 5.0))
SHADOW_ALIAS = os.environ.get("SHADOW_ALIAS", "")  # e.g. "challenger": scored on live traffic, never acted on
ALIAS_POLL_S = float(os.environ.get("ALIAS_POLL_S", 15))  # each worker re-checks the registry aliases this often
N_MONITORED = 20  # top features by gain that are echoed into scored-events for drift monitoring

LAT_BUCKETS = (.001, .002, .003, .005, .0075, .01, .015, .02, .03, .04, .05, .065, .08, .1, .15, .25, .5, 1)
REQUEST_SECONDS = Histogram("fraud_score_request_seconds", "/score handler time", buckets=LAT_BUCKETS)
STAGE_SECONDS = Histogram("fraud_score_stage_seconds", "Time per scoring stage", ["stage"],
                          buckets=LAT_BUCKETS)
SCORE_VALUE = Histogram("fraud_score_value", "Model probability",
                        buckets=(1e-5, 1e-4, 1e-3, .002, .005, .01, .02, .05, .1, .25, .5, .75, .9, .99, 1))
DECISIONS = Counter("fraud_decisions_total", "Decisions", ["decision", "model_version"])
FEATURE_MISSING = Counter("fraud_feature_missing_total", "Entities with no online features", ["entity"])
ERRORS = Counter("fraud_score_errors_total", "Failed /score calls", ["kind"])
SHADOW_DECISIONS = Counter("fraud_shadow_decisions_total", "Shadow model decisions (not acted on)",
                           ["decision", "model_version"])
SHADOW_DISAGREE = Counter("fraud_shadow_disagreements_total", "Shadow flag differs from the champion's")
SHADOW_ERRORS = Counter("fraud_shadow_errors_total", "Shadow scoring failures (never fail the request)", ["kind"])


class Transaction(BaseModel):
    """Raw IEEE-CIS transaction. Any other column of the dataset may be passed; absent ones are missing."""
    model_config = ConfigDict(extra="allow")
    TransactionID: int
    TransactionDT: int
    TransactionAmt: float = Field(gt=0)


class ScoreResponse(BaseModel):
    transaction_id: int
    score: float
    flagged: bool
    threshold: float
    model_version: str
    missing_entities: list[str]


def build_scorer() -> Scorer:
    booster, threshold, version = load_alias(ALIAS)
    return Scorer(booster, threshold, version, FeastFetcher())


def build_shadow(champion: Scorer) -> Scorer | None:
    """Scorer for the SHADOW_ALIAS model, reusing the champion's feature fetcher. None if shadow mode is off
    or the alias does not exist yet: shadow mode must never stop the champion from serving."""
    if not SHADOW_ALIAS:
        return None
    try:
        booster, threshold, version = load_alias(SHADOW_ALIAS)
    except Exception:
        log.warning("shadow alias %r not available; shadow scoring off", SHADOW_ALIAS, exc_info=True)
        return None
    if str(version) == champion.version:  # MLflow returns the version as an int, Scorer stores a str
        return None  # right after a promotion the challenger alias still points at the champion
    log.info("shadow model v%s threshold=%g", version, threshold)
    return Scorer(booster, threshold, version, champion.fetch)


def build_publisher(scorer: Scorer):
    if not PUBLISH:
        return NullPublisher()
    return EventPublisher(BOOTSTRAP, monitored_columns(scorer.booster, N_MONITORED))


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not WATCHDOG.is_alive():
        WATCHDOG.start()
    app.state.pool = ThreadPoolExecutor(SCORE_THREADS, thread_name_prefix="score")
    app.state.explain_pool = None
    app.state.explain_slots = asyncio.Semaphore(EXPLAIN_SLOTS)
    app.state.scorer = build_scorer()
    app.state.shadow = build_shadow(app.state.scorer)
    app.state.publisher = build_publisher(app.state.scorer)
    if EXPLAIN_EAGER:
        _explain_pool(warm=True)
    log.info("serving model v%s threshold=%g", app.state.scorer.version, app.state.scorer.threshold)
    watcher = asyncio.create_task(_watch_aliases()) if ALIAS_POLL_S > 0 else None
    yield
    if watcher:
        watcher.cancel()
    app.state.publisher.close()
    app.state.pool.shutdown(wait=False)
    if app.state.explain_pool is not None:
        app.state.explain_pool.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="fraud-stream scoring", lifespan=lifespan)


@app.get("/health")
def health():
    s, sh = app.state.scorer, app.state.shadow
    return {"status": "ok", "alias": ALIAS, "model_version": s.version, "threshold": s.threshold,
            "features": len(s.names), "events_dropped": app.state.publisher.dropped,
            "shadow_version": sh.version if sh else None, "pid": os.getpid()}


class _Watchdog(threading.Thread):
    """Dumps every thread's stack if requests are in flight but none has finished for STALL_S seconds,
    so a frozen worker leaves evidence (stall_dump.log) instead of just timing out clients."""

    def __init__(self, stall_s: float = 10.0):
        super().__init__(daemon=True, name="stall-watchdog")
        self.stall_s, self.inflight, self.last_done = stall_s, 0, time.monotonic()

    def run(self):
        while True:
            time.sleep(2)
            if self.inflight > 0 and time.monotonic() - self.last_done > self.stall_s:
                with open(os.environ.get("FRAUD_STALL_DUMP", "stall_dump.log"), "a") as f:
                    f.write(f"\n=== pid {os.getpid()} {time.strftime('%H:%M:%S')}: {self.inflight} requests "
                            f"in flight, none finished for {time.monotonic() - self.last_done:.0f}s ===\n")
                    f.flush()
                    faulthandler.dump_traceback(file=f, all_threads=True)
                self.last_done = time.monotonic() + 30  # do not dump again for a while


WATCHDOG = _Watchdog()


def _score_and_record(raw: dict) -> dict:
    """The whole blocking part of /score (lookup, model, metrics, event) in one worker-thread hop."""
    WATCHDOG.inflight += 1
    try:
        return _score_and_record_inner(raw)
    finally:
        WATCHDOG.inflight -= 1
        WATCHDOG.last_done = time.monotonic()


def _score_shadow(raw: dict, r: dict) -> None:
    """Score the shadow model on the champion's features and attach the result to `r["shadow"]`. The decision
    returned to the client is always the champion's; a shadow failure is counted, not raised."""
    shadow = app.state.shadow
    if shadow is None:
        return
    try:
        s = shadow.score(raw, given=(r["features"], r["missing_entities"]))
    except Exception as exc:
        SHADOW_ERRORS.labels(type(exc).__name__).inc()
        log.warning("shadow scoring failed: %s", exc)
        return
    r["shadow"] = {"score": s["score"], "flagged": s["flagged"], "threshold": s["threshold"],
                   "model_version": s["model_version"]}
    r["timings_ms"]["shadow"] = s["timings_ms"]["encode"] + s["timings_ms"]["predict"]
    SHADOW_DECISIONS.labels("flag" if s["flagged"] else "allow", s["model_version"]).inc()
    if s["flagged"] != r["flagged"]:
        SHADOW_DISAGREE.inc()


def _score_and_record_inner(raw: dict) -> dict:
    r = app.state.scorer.score(raw)
    _score_shadow(raw, r)
    for stage, ms in r["timings_ms"].items():
        STAGE_SECONDS.labels(stage).observe(ms / 1e3)
    for e in r["missing_entities"]:
        FEATURE_MISSING.labels(e).inc()
    SCORE_VALUE.observe(r["score"])
    DECISIONS.labels("flag" if r["flagged"] else "allow", r["model_version"]).inc()
    app.state.publisher.publish(raw, r)
    return r


@app.post("/score", response_model=ScoreResponse)
async def score(txn: Transaction):
    # One hand-off to a dedicated pool instead of several per request: under load every extra
    # thread hop is another wait for the GIL, which dominated latency in measurements (see README).
    t0 = time.perf_counter()
    try:
        r = await asyncio.get_running_loop().run_in_executor(app.state.pool, _score_and_record,
                                                            txn.model_dump())
    except Exception as exc:  # feature store down etc.: fail loudly, never guess a score
        ERRORS.labels(type(exc).__name__).inc()
        log.exception("scoring failed")
        raise HTTPException(503, f"scoring unavailable: {type(exc).__name__}") from exc
    REQUEST_SECONDS.observe(time.perf_counter() - t0)
    return ScoreResponse(transaction_id=txn.TransactionID, score=r["score"], flagged=r["flagged"],
                         threshold=r["threshold"], model_version=r["model_version"],
                         missing_entities=r["missing_entities"])


def _explain_pool(warm: bool = False) -> ProcessPoolExecutor:
    """SHAP process pool for the current model. Built at startup and after /reload (saving the model and
    spawning the process take seconds, which must not happen inside a request)."""
    if app.state.explain_pool is None:
        path = Path(tempfile.mkdtemp(prefix="fraud_explain_")) / "model.txt"
        app.state.scorer.booster.save_model(str(path))
        app.state.explain_pool = ProcessPoolExecutor(EXPLAIN_SLOTS, initializer=explain_worker.init,
                                                     initargs=(str(path),))
        if warm:
            app.state.explain_pool.submit(explain_worker.ping).result(timeout=60)
    return app.state.explain_pool


@app.post("/explain")
async def explain(txn: Transaction, k: int = 3):
    """Top-k SHAP reasons for one transaction. Slow (~0.4 s of CPU): call it for flagged transactions only.

    Runs in separate processes with capped concurrency; excess requests wait briefly, then get a 503
    instead of an unbounded queue."""
    loop = asyncio.get_running_loop()
    try:
        async with asyncio.timeout(EXPLAIN_WAIT_S):
            async with app.state.explain_slots:
                row = await loop.run_in_executor(app.state.pool, app.state.scorer.explain_row, txn.model_dump())
                reasons = await loop.run_in_executor(_explain_pool(), explain_worker.shap, row, k)
    except TimeoutError:
        raise HTTPException(503, "explanations busy, retry shortly") from None
    return {"transaction_id": txn.TransactionID, "reasons": reasons}


def _alias_drift() -> bool:
    """True if the registry's champion/shadow aliases no longer match what this worker is serving."""
    champ = alias_version(ALIAS)
    want_shadow = alias_version(SHADOW_ALIAS) if SHADOW_ALIAS else None
    if want_shadow == champ:
        want_shadow = None
    have = app.state.shadow.version if app.state.shadow else None
    return champ is not None and (champ != app.state.scorer.version or want_shadow != have)


async def _watch_aliases():
    """POST /reload only reaches the one worker that handles it, but a promotion must reach all of them. Every
    worker therefore polls the registry and reloads itself when an alias moved."""
    loop = asyncio.get_running_loop()
    while True:
        await asyncio.sleep(ALIAS_POLL_S)
        try:
            if await loop.run_in_executor(None, _alias_drift):
                log.info("registry alias changed; reloading")
                await loop.run_in_executor(None, reload)
        except Exception:
            log.warning("alias check failed; keeping the current models", exc_info=True)


@app.post("/reload")
def reload():
    """Re-read the registry alias (after a promotion) and swap the model in place."""
    new = build_scorer()
    old, app.state.scorer = app.state.scorer.version, new
    app.state.shadow = build_shadow(new)  # after a promotion the old challenger is the champion: shadow off
    if app.state.explain_pool is not None:  # SHAP processes hold the old model: drop them, rebuilt on demand
        app.state.explain_pool.shutdown(wait=False, cancel_futures=True)
        app.state.explain_pool = None
    if EXPLAIN_EAGER:
        _explain_pool(warm=True)
    return {"previous_version": old, "model_version": new.version, "threshold": new.threshold,
            "shadow_version": app.state.shadow.version if app.state.shadow else None}


@app.get("/metrics")
def metrics():
    # Feast sets PROMETHEUS_MULTIPROC_DIR itself on import, so use our own flag (set by serving.run)
    if os.environ.get("FRAUD_PROM_MULTIPROC") == "1":  # several worker processes: aggregate their files
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
