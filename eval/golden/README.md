# Golden set

| File | Committed? | Contents |
|---|---|---|
| `candidates.jsonl` | **no** (gitignored) | Generated Q/A candidates; may quote near-verbatim passages. |
| `golden.jsonl` | yes | Owner-verified questions, **paraphrased** key points, source chunk *hashes*, type, chapter. |
| `golden.verified.json` | yes | Stamp: sha256 of `golden.jsonl`, counts, date. The harness refuses to run if it is missing or stale. |
| `judge_labels.jsonl` | yes | 30 owner-labeled (question, answer, context-hashes, human_pass) rows for judge validation. |

Workflow: `make golden-generate` → `make golden-verify` (interactive: accept / edit / reject each pair with its
source chunk shown) → `make eval`. Chunk hashes rather than ids label sources so the set survives re-ingestion;
`mlsys_eval.harness.resolve_hashes` maps them to current ids.
