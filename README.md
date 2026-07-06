# ingest-core

The single home for reusable market-data acquisition plumbing shared by **Hermes** and **Hephaestus** (feature `003-ingest-core-orchestration`).

## Contents

| Module | Purpose |
|---|---|
| `ingest_core.clients.eodhd` | EODHD HTTP client: URL builders, paginated fetch with 429/5xx backoff, quota status (promoted from Hephaestus) |
| `ingest_core.clients.barchart` | Barchart acquisition: most-active options fetch, Russell-1000 core-api fetch (from Hermes), most-active CSV reader (from Hephaestus) |
| `ingest_core.ratelimit` | In-process `RateLimiter` — per-app, no cross-process state |
| `ingest_core.calendar` | Trading-calendar helpers: trading-days-between, missing-day gap detection, range collapse (from Hermes), NYSE session/offset utilities (from Hephaestus) |
| `ingest_core.retry` | Orchestration-agnostic retry/backoff wrapper |
| `ingest_core.backfill` | Backfill plumbing: thread-safe `Budget`, windowed range fetch with auto-split on the offset cap, pre-history walk (stop at first empty week), live quota sizing + reset probe |
| `ingest_core.types` | Shared DTOs + `EODHDLimitError` |

## Hard boundaries (FR-003/FR-004)

- **No Prefect** (or any scheduler) — orchestration lives in the consuming apps / the `orchestration/` conductor.
- **No storage** — no SQLAlchemy engines, no Parquet writes. The library returns data; apps own loaders/staging/schemas.
- **No project domain logic** — membership sync, grade scrapes, ticker upserts, etc. stay app-side.

## Install (editable, both consumers)

```bash
# from within Hermes/ or Hephaestus/ venv
pip install -e ../ingest-core
```

## Tests

```bash
pytest tests/unit -q     # hermetic: mocked HTTP, deterministic calendar
```
