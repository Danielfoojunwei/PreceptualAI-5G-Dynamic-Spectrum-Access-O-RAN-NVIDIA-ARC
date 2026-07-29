# Evidence Store Backends

The evidence store persists every `DecisionRecord` behind a SHA-256 hash
chain. Three interchangeable backends implement the same `EvidenceStore`
ABC (`src/horizon_ric/evidence/store.py`); the Helm chart's TimescaleDB
StatefulSet is now a real, first-class backend rather than an unused
deployment.

## Backend matrix

| Backend | Class | Target | Storage | Concurrency guard | Use |
|---|---|---|---|---|---|
| JSONL | `JsonlEvidenceStore` | any path (default) | one canonical-JSON line + hash per record | in-process `threading.Lock` | dev, single-pod audit file |
| SQLite | `SqliteEvidenceStore` | `*.db` / `*.sqlite` / `sqlite://` URL | `decisions` table (sqlalchemy) | in-process `threading.Lock` | single-node queries, REST API |
| PostgreSQL / TimescaleDB | `PostgresEvidenceStore` | `postgres://` / `postgresql://` DSN | `evidence_records` table (JSONB), auto-hypertable on TimescaleDB | **`pg_advisory_xact_lock` per tenant** — safe across threads, connections, and processes | production (Helm `timescaledb` StatefulSet) |

## The `HORIZON_EVIDENCE_DSN` contract

`open_evidence_store(target: str | Path) -> EvidenceStore` dispatches:

1. `postgres://…` or `postgresql://…` → `PostgresEvidenceStore(dsn)`
   (DSN passed straight to psycopg/libpq; connection is **lazy** — the
   store connects and auto-creates schema on first use, not on
   construction).
2. `sqlite://…` → `SqliteEvidenceStore(url)`.
3. Path suffix `.db` / `.sqlite` / `.sqlite3` → `SqliteEvidenceStore`.
4. Anything else → `JsonlEvidenceStore(path)`.

The rApp daemon reads `HORIZON_EVIDENCE_DSN` and falls back to its local
audit path, so pointing the env var at the in-cluster DSN
(`postgresql://horizon:…@timescaledb:5432/decisions`) switches the pod to
the durable backend with no code change.

## Chain semantics (identical in all three backends)

```
sha256_curr = sha256(prev_sha256 || canonical_json(record))
```

* `canonical_json` = pydantic dump → `json.dumps(sort_keys=True,
  separators=(",", ":"))`.
* One independent chain **per tenant** (`tenant_id` stamped from the
  active `TenantScope` when the record has none; `_unscoped_` sentinel
  otherwise). First record of each tenant chains from the all-zero hash.
* `verify()` walks all chains, returns the global index of the first
  broken record or `-1`; `verify_tenant(t)` walks one chain and returns a
  per-tenant index. Tampering tenant A never breaks tenant B's verify.
* Iteration/`len` inside a `TenantScope` see only that tenant's rows.

### Postgres specifics

* Table (auto-created, `CREATE TABLE IF NOT EXISTS` + index on
  `(tenant_id, id)`):

  ```sql
  CREATE TABLE evidence_records (
      id          BIGINT GENERATED ALWAYS AS IDENTITY,
      tenant_id   TEXT NOT NULL,
      ts          TIMESTAMPTZ NOT NULL,
      decision_id TEXT NOT NULL,
      record      JSONB NOT NULL,
      hash        TEXT NOT NULL
  );
  ```

  No PRIMARY KEY on purpose: TimescaleDB rejects `create_hypertable` when
  a unique index omits the partition column (`ts`); append order is the
  `id` identity column and integrity is the hash chain itself.
* The read-prev-hash→insert critical section runs under
  `pg_advisory_xact_lock(0x48524943, hashtext(tenant_id))` inside the
  insert transaction, so concurrent appends from separate
  connections/pods cannot fork a tenant's chain. Schema DDL takes the
  same lock with key 0 (plain `CREATE TABLE IF NOT EXISTS` is not
  race-free — two first-connectors can collide in `pg_class`; the live
  concurrency test caught exactly this).
* On TimescaleDB, first connect best-efforts
  `SELECT create_hypertable('evidence_records','ts', if_not_exists=>TRUE,
  migrate_data=>TRUE)` in a try/except; plain PostgreSQL rolls it back
  and keeps working.

## Live test evidence (this machine, 2026-07-27)

`tests/test_evidence_postgres_live.py` boots a real
`timescale/timescaledb:2.16.1-pg16` container (`docker run -d
--network=host … -c port=55432`) — no mocks — and skips only when the
docker daemon is unavailable. `tests/test_evidence_store_factory.py`
covers factory dispatch without docker.

```
$ /home/user/venv/bin/python -m pytest tests/test_evidence_postgres_live.py \
      tests/test_evidence_store_factory.py -q
....................                                                     [100%]
20 passed
```

9 live tests, all green against the real container: 5 appends across 2
tenants verify to `-1`; a `jsonb_set` UPDATE forging one row makes
`verify()` return exactly that row's index (and only that tenant's
chain breaks); `__len__`/`__iter__` round-trip a fully-populated
`DecisionRecord` byte-identically (canonical JSON equality through
JSONB); 20 concurrent appends from 4 threads *each with its own
connection* keep valid chains (the advisory lock, not the client lock,
is what protects this); `timescaledb_information.hypertables` confirms
`evidence_records` became a hypertable; the factory opens the live DSN.

## MinIO backup/restore proof (this machine, 2026-07-27)

Real containers, real `aws` CLI (awscli 1.45.56 installed in an isolated
venv — the script's postgres path shells out to `aws s3`), real
`pg_dump`/`pg_restore` 16.13, `cosign` absent so the script's documented
HMAC fallback signed the blobs:

```
$ docker run -d --network=host --name horizon-dr-pg \
    -e POSTGRES_PASSWORD=horizon-e2e -e POSTGRES_DB=decisions \
    timescale/timescaledb:2.16.1-pg16 -c port=55432
$ docker run -d --network=host --name horizon-dr-minio \
    -e MINIO_ROOT_USER=horizon -e MINIO_ROOT_PASSWORD=horizon-e2e \
    minio/minio server /data --address :59000 --console-address :59001
# 6 real DecisionRecords appended via open_evidence_store(DSN):
#   rows: 6 verify: -1
$ aws --endpoint-url http://127.0.0.1:59000 s3 mb s3://horizon-evidence-archive
$ HORIZON_DR_BACKEND=postgres \
  HORIZON_DR_DB_URL="postgres://postgres:horizon-e2e@127.0.0.1:55432/decisions" \
  HORIZON_DR_S3_ENDPOINT=http://127.0.0.1:59000 \
  bash scripts/backup_and_restore.sh backup
[backup-restore] dumped db -> /tmp/horizon-dump.iUa36j (15445 bytes)
[backup-restore] WARNING: cosign unavailable; wrote HMAC fallback signature ...
[backup-restore] stored object decisions-20260727T061135Z.dump
[backup-restore] stored object decisions-20260727T061135Z.dump.sig
[backup-restore] backup complete: decisions-20260727T061135Z.dump
$ aws --endpoint-url http://127.0.0.1:59000 s3 ls s3://horizon-evidence-archive/decisions/
2026-07-27 06:11:36      15445 decisions-20260727T061135Z.dump
2026-07-27 06:11:36         32 decisions-20260727T061135Z.dump.sig
$ psql "$HORIZON_DR_DB_URL" -c "DROP TABLE evidence_records;"   # simulate loss
$ bash scripts/backup_and_restore.sh restore decisions-20260727T061135Z.dump
[backup-restore] fetched object decisions-20260727T061135Z.dump
[backup-restore] HMAC verified /tmp/horizon-restore.uBpk4w
[backup-restore] restore complete from decisions-20260727T061135Z.dump
# post-restore, via PostgresEvidenceStore:
#   rows after restore: 6, verify(): -1, both tenants -1, hypertable: yes
```

### Backup-script defect found and fixed

`restore_database()` in `scripts/backup_and_restore.sh` ran plain
`pg_restore --no-owner --clean --if-exists` for every postgres target.
Against a TimescaleDB source dump this **failed live** (35 errors): the
`--clean` cascade drops the `timescaledb` extension, the catalog cannot
be recreated mid-restore, and the target is left with the extension gone
and **zero** restored rows. Fixed (in-script, postgres branch only) with
the TimescaleDB-documented sequence — `CREATE EXTENSION IF NOT EXISTS
timescaledb` → `SELECT timescaledb_pre_restore()` → `pg_restore
--exit-on-error --use-list <TOC minus the dump's own CREATE EXTENSION
entry>` → `SELECT timescaledb_post_restore()` — applied only when the
target server offers the timescaledb extension; plain-Postgres targets
keep the original `--clean` path. The TimescaleDB path restores into a
fresh/prepared database (Timescale's own requirement; matches
`deploy/DR_PLAN.md`'s fresh-volume restore flow). Verified live above.

## Honest scope notes

* `PostgresEvidenceStore.append` re-reads the tenant's last hash inside
  each transaction (no cache) — correct under multi-writer, ~1 extra
  indexed SELECT per append; fine at rApp decision rates.
* One psycopg connection per store instance, guarded by a
  `threading.Lock`; cross-instance/process safety comes from the
  advisory lock. No pooling/reconnect-backoff — a dropped connection is
  reopened lazily on the next call.
* JSONB numeric round-trip relies on shortest-repr floats (Python
  `repr` ↔ Postgres `numeric`); NaN/Inf are not valid JSON and are
  rejected at insert, same as the JSONL backend.
* The `decision_id` uniqueness constraint of the SQLite backend is not
  replicated (TimescaleDB forbids unique indexes without the partition
  column); duplicates are chain-visible but not DB-rejected.
* The MinIO drill exercised `backup`, `list`, and `restore` with the
  HMAC fallback signature (no `cosign` on this host); the cosign path is
  unchanged and still gated by `HORIZON_DR_REQUIRE_COSIGN=1`.
* `scripts/backup_and_restore.sh` `truncate_database()` still targets
  the legacy `decisions` table; it is dead code (never reached from the
  CLI) and was left untouched — flagged here for the owner of that
  script's test drill.
* Wiring the daemon to `HORIZON_EVIDENCE_DSN` is owned by a parallel
  change (`scripts/run_horizon_rapp.py`); this document covers the store
  and DR layers only.
