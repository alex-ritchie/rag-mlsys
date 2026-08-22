-- Schema (spec §5.2 + §5.5). Applied idempotently by `python -m mlsys_common.db migrate`.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
  id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  volume        SMALLINT NOT NULL,
  chapter_num   SMALLINT NOT NULL,
  chapter_title TEXT NOT NULL,
  section_path  TEXT[] NOT NULL,
  heading_path  TEXT NOT NULL,
  source_file   TEXT NOT NULL,
  char_start    INT NOT NULL,
  char_end      INT NOT NULL,
  token_count   INT NOT NULL,
  commit_sha    TEXT NOT NULL,
  content_hash  TEXT NOT NULL,
  oversize      BOOLEAN NOT NULL DEFAULT FALSE,
  text          TEXT NOT NULL,
  embedding     vector(1024),
  fts           tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
);
CREATE UNIQUE INDEX IF NOT EXISTS chunks_content_hash_idx ON chunks (content_hash);
CREATE INDEX IF NOT EXISTS chunks_fts_idx ON chunks USING gin (fts);
CREATE INDEX IF NOT EXISTS chunks_chapter_idx ON chunks (volume, chapter_num);
-- HNSW index is created by `make index` after embeddings exist (building it on an
-- empty table and then bulk-inserting is much slower than build-after-load).

CREATE TABLE IF NOT EXISTS ingest_runs (
  id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  commit_sha    TEXT NOT NULL,
  started_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  finished_at   TIMESTAMPTZ,
  chunks_added  INT NOT NULL DEFAULT 0,
  chunks_removed INT NOT NULL DEFAULT 0,
  chunks_unchanged INT NOT NULL DEFAULT 0,
  notes         TEXT
);

-- Per-query log (spec §5.5). No user identifiers.
CREATE TABLE IF NOT EXISTS query_log (
  id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  question_hash   TEXT NOT NULL,
  question        TEXT NOT NULL,
  started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  finished_at     TIMESTAMPTZ,
  profile         TEXT NOT NULL,
  model           TEXT NOT NULL,
  prompt_version  TEXT NOT NULL,
  retrieval_mode  TEXT NOT NULL,
  embed_ms        REAL, retrieve_ms REAL, rerank_ms REAL, ttft_ms REAL, generate_ms REAL, total_ms REAL,
  fused_ids       BIGINT[] NOT NULL DEFAULT '{}',
  fused_scores    REAL[]   NOT NULL DEFAULT '{}',
  reranked_ids    BIGINT[] NOT NULL DEFAULT '{}',
  rerank_scores   REAL[]   NOT NULL DEFAULT '{}',
  prompt_tokens   INT, completion_tokens INT,
  answer          TEXT,
  abstained       BOOLEAN,
  error           TEXT
);
CREATE INDEX IF NOT EXISTS query_log_started_idx ON query_log (started_at DESC);

-- Nightly groundedness (drift CronJob, spec §5.8) writes here; /metrics reads it.
CREATE TABLE IF NOT EXISTS judge_scores (
  id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  query_log_id  BIGINT REFERENCES query_log(id) ON DELETE CASCADE,
  judged_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
  judge_model   TEXT NOT NULL,
  faithfulness  REAL, relevance REAL, groundedness REAL,
  flagged       BOOLEAN NOT NULL DEFAULT FALSE,
  rationale     TEXT
);

-- Demo-profile cost controls (spec §5.11). Keyed by a salted IP hash, never the IP.
CREATE TABLE IF NOT EXISTS demo_requests (
  id         BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  ip_hash    TEXT NOT NULL,
  at         TIMESTAMPTZ NOT NULL DEFAULT now(),
  cost_usd   REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS demo_requests_ip_idx ON demo_requests (ip_hash, at DESC);
CREATE INDEX IF NOT EXISTS demo_requests_at_idx ON demo_requests (at DESC);
