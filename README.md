# MREMA starter: PostgreSQL + CSV import

This is a starter scaffold, not a replacement for your existing `query_system`.
Copy/merge the files into your repository after reviewing the existing Compose and Dockerfiles.

## 1. Setup
```bash
cp .env.example .env
# Edit POSTGRES_PASSWORD in .env for your local environment.
docker compose up --build
```

API docs: http://localhost:8000/docs  
Health: http://localhost:8000/health

## 2. Import a CSV
```bash
curl -X POST http://localhost:8000/api/datasets/import \
  -F "file=@./sample.csv"
```

List imported datasets:
```bash
curl http://localhost:8000/api/datasets
```

Preview the first 20 rows (replace 1 with returned id):
```bash
curl "http://localhost:8000/api/datasets/1/preview?limit=20&offset=0"
```

## 3. Run a frontend

Requires Streamlit:
```bash
pip install streamlit
```

Then from the repo root:
```bash
streamlit run ./frontend/main.py
```

## 4. Important notes before using with real hackathon data
- This initial implementation reads each CSV into a pandas DataFrame. It is appropriate for a first integration test, not an unlimited-size ingestion service.
- The import endpoint has a configurable upload limit but currently buffers the upload in memory. Use streaming/chunked ingestion for larger files.
- Pandas infers SQL column types. Inspect and harden this for identifiers with leading zeros, mixed-type columns, locale-specific decimals, unusual encodings and malformed rows.
- A production implementation should use Alembic migrations, structured logging, request timeouts, stronger error handling and integration tests.
- The starter intentionally uses generated table names and normalized column names; the original header is kept in column profile metadata.
- The example `.env` credentials are development-only. Never commit real secrets.
- `query_system.QueryEngine` still executes against its in-memory `sources` mapping. This scaffold does not automatically translate your custom query-step JSON to SQL. Integrate that separately, or implement a validated SQL compiler.
