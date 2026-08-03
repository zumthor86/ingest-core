# ingest-core

The single home for reusable market-data acquisition plumbing shared by **Hermes** and **Hephaestus** (feature `003-ingest-core-orchestration`). Both consumers editable-install it as a build-time dependency; it holds no orchestration or storage (storage is its sibling library [`store-core`](../store-core/README.md)). See [`../ARCHITECTURE.md`](../ARCHITECTURE.md) for how it fits the daily pipeline.

## Contents

| Module | Purpose |
|---|---|
| `ingest_core.clients.eodhd` | EODHD HTTP client: options URL builders + paginated fetch with 429/5xx backoff + quota status (from Hephaestus); EOD / bulk / intraday price fetchers and exchange-metadata fetchers (`fetch_exchange_catalog`, `fetch_exchange_symbol_list`) (from Hermes) |
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

## Changelog

Dated, one-line entries for changes that affect consumers — new modules, client
contract or backoff-behavior changes, breaking behavior. Keep entries short;
`git log` has the detail. Update this **in the same change** that touches this
library, and mention it in whichever consuming project's `CLAUDE.md` you're also
updating.

- **2026-07-06** (`d66f110`) — Initial extraction from Hermes + Hephaestus: `clients.eodhd`, `clients.barchart`, `ratelimit`, `calendar`, `retry`, `backfill`, `types`.
- **2026-07-06** — `ingest_core.calendar` gains `last_completed_session()`: resolves "latest trading day" as the most recently **closed** session, not the still-open "today" — a mid-session intraday trigger of a coverage/readiness gate keyed on "today" previously read 0% and failed until the market closed. `clients.barchart`'s `_parse_barchart_value`/`_parse_barchart_date` are renamed to public (`parse_barchart_value`/`parse_barchart_date`).
