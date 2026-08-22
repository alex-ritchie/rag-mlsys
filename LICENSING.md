# Licensing and compliance

This document records how the project complies with the licenses of everything it touches. It is a
hard constraint on every milestone (design doc §7), not an afterthought.

## 1. The two licenses in play

| Thing | License | Where it applies |
|---|---|---|
| *Machine Learning Systems* (text, figures, `.qmd` sources) by Vijay Janapa Reddi | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) (per `LICENSE.md` in [harvard-edge/cs249r_book](https://github.com/harvard-edge/cs249r_book)) | Anything derived from the book: chunks, embeddings, indexes, retrieved passages, golden-set *source* passages |
| This repository's code, configs, manifests, docs | [MIT](LICENSE) | Everything committed here |

## 2. Attribution (BY) — in three places

1. **README** — header block credits the author, the book, mlsysbook.ai, and the license.
2. **App UI** — the About page carries the full attribution block and the index commit SHA; the
   footer on every page states the book, author, and license (`frontend/src/App.tsx`,
   `frontend/src/pages/AboutPage.tsx`).
3. **Per-answer citations** — every answer cites the chapter/section it drew from (`[n]` chips mapped
   to `heading_path`), and the retrieval inspector shows the passage with its provenance.

## 3. Never commit book content or derived indexes

- The book is fetched at a pinned commit into `data/` which is **gitignored**
  (`config/ingest.yaml` → `data/book`). Chunks, embeddings, HNSW indexes, model caches: all under `data/`.
- `.gitignore` blocks `data/`, `*.qmd` (except synthetic fixtures under `ingest/tests/fixtures/`),
  and model artifacts. `scripts/guard_content.py` runs in pre-commit and CI and fails on any of those,
  on any file > 20 MB, and on tripwire strings that occur in real chapter sources.
- Test fixtures are synthetic ("Widget Systems", "Gear Trains"). Eval artifacts contain questions,
  scores, and **paraphrased** key points written by the owner during verification — never extracted
  passages. The candidates file produced by generation (`eval/golden/candidates.jsonl`) is gitignored
  because it may quote near-verbatim text.
- We ship the *pipeline*. Every user builds their own index from the source repository.

## 4. Non-commercial (NC)

The hosted demo is free, non-monetized, has no accounts, no paywalls, no paid tiers, and serves only
to let a visitor try the system without a GPU. Any future change to that must revisit this document
and the license.

## 5. ShareAlike (SA)

No derived artifact is published. The Supabase table used by the demo holds chunks and embeddings for
the demo's own retrieval; it is private (service-role access only, never exposed as an API), which is
private hosting for the demo's own use, not distribution. If a derived artifact is ever published, it
will be licensed CC BY-NC-SA 4.0.

## 6. Site terms check (performed during M1, 2026-08-21)

The content is obtained from the GitHub repository (CC BY-NC-SA 4.0), not by crawling mlsysbook.ai.
For completeness, `https://mlsysbook.ai/robots.txt` was checked on 2026-08-21:

```
User-agent: *
Content-Signal: search=yes,ai-train=no,use=reference
Allow: /
```
plus `Disallow: /` for a list of named AI crawlers (GPTBot, ClaudeBot, CCBot, Google-Extended, …).
No `ai.txt` (404). Reading of the Content-Signals: `ai-train=no` (we do no training or fine-tuning —
out of scope by design), `search=yes`, and no `ai-input` signal, which per the file's own definition
"neither grants nor restricts permission" for retrieval-augmented use. The project does not crawl the
site at all; the crawler disallow list therefore does not apply to it. The governing terms remain the
repository license.

## 7. Models

| Model | License |
|---|---|
| BAAI/bge-m3, BAAI/bge-reranker-v2-m3 | MIT |
| Qwen3.8-27B (and the W4A16 / GGUF community quantizations), Qwen3.5-9B, Qwen3.6-35B-A3B | Apache 2.0 |
| Claude Haiku (demo generation, judge) | Anthropic commercial terms; API key is server-side only |
