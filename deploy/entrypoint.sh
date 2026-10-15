#!/bin/sh
set -e
# Railway's Redis plugin exposes REDIS_URL (redis://user:pass@host:port); Feast wants "host:port,password=...".
if [ -z "$REDIS_CONNECTION_STRING" ] && [ -n "$REDIS_URL" ]; then
  REDIS_CONNECTION_STRING=$(python - <<'PY'
import os
from urllib.parse import urlparse
u = urlparse(os.environ["REDIS_URL"])
s = f"{u.hostname}:{u.port or 6379}"
if u.password:
    s += f",password={u.password}"
if u.scheme == "rediss":
    s += ",ssl=true"
print(s)
PY
)
  export REDIS_CONNECTION_STRING
fi
python -m src.serving.seed
exec python -m src.serving.run --workers "${WORKERS:-2}" --port "${PORT:-8000}"
