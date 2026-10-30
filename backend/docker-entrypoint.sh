#!/bin/sh
set -e

if [ -n "$DATABASE_URL" ]; then
  echo "Waiting for postgres..."
  python - <<'PY'
import os, socket, sys, time
from urllib.parse import urlparse

url = os.environ.get("DATABASE_URL", "")
if not url.startswith("postgresql"):
    sys.exit(0)

parsed = urlparse(url.replace("postgresql+psycopg", "postgresql"))
host, port = parsed.hostname or "postgres", parsed.port or 5432

for attempt in range(60):
    try:
        with socket.create_connection((host, port), timeout=2):
            print(f"Postgres at {host}:{port} is up")
            break
    except OSError:
        print(f"Waiting for postgres at {host}:{port}... ({attempt + 1}/60)")
        time.sleep(1)
else:
    print("Timeout waiting for postgres", file=sys.stderr)
    sys.exit(1)
PY
fi

exec "$@"
