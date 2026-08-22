import { useEffect, useState } from 'react'
import { API_BASE } from '../lib/config'
// The committed report is bundled at build time (static data, no backend dependency — spec §5.10 #4).
import bundled from '../../../eval/results/latest/report.json'

type Row = { config: string; recall_at_5: number; recall_at_10: number; recall_at_30: number; mrr: number; n: number }
interface Report {
  run_id: string
  created_at: string
  model: string
  prompt_version: string
  judge_model: string
  index_commit_sha: string | null
  golden_count: number
  golden_by_type: Record<string, number>
  retrieval: Row[]
  generation: null | {
    model: string
    n: number
    faithfulness_mean: number
    faithfulness_pass_rate: number
    relevance_mean: number
    relevance_pass_rate: number
    groundedness_mean: number
    groundedness_pass_rate: number
    abstention: Record<string, number>
    by_type: Record<string, Record<string, number>>
    latency_ms: Record<string, number>
  }
  judge_agreement: null | { n: number; percent_agreement: number; cohen_kappa: number; trusted: boolean }
  models?: { model: string; faithfulness_mean: number; relevance_mean: number; groundedness_mean: number; abstention_f1: number }[]
}

const pct = (x: number) => `${(100 * x).toFixed(1)}%`

function Card({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="card">
      <div className="muted text-xs">{label}</div>
      <div className="text-2xl font-semibold">{value}</div>
      {sub && <div className="muted text-xs">{sub}</div>}
    </div>
  )
}

export default function EvalPage() {
  const [r, setR] = useState<Report>(bundled as unknown as Report)
  useEffect(() => {
    // prefer a fresher report from the gateway when one is available; the bundled copy is the fallback
    fetch(`${API_BASE}/api/eval/summary`)
      .then((x) => (x.ok ? x.json() : null))
      .then((j) => j && setR(j as Report))
      .catch(() => {})
  }, [])
  const g = r.generation
  return (
    <div className="space-y-6">
      <div className="muted text-sm">
        run <code className="font-mono">{r.run_id}</code> · model {r.model} · prompt {r.prompt_version} · judge {r.judge_model} · {r.golden_count} golden questions (
        {Object.entries(r.golden_by_type)
          .map(([k, v]) => `${v} ${k}`)
          .join(', ')}
        )
      </div>
      {g && (
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <Card label="faithfulness" value={g.faithfulness_mean.toFixed(2)} sub={`pass ${pct(g.faithfulness_pass_rate)}`} />
          <Card label="answer relevance" value={g.relevance_mean.toFixed(2)} sub={`pass ${pct(g.relevance_pass_rate)}`} />
          <Card label="groundedness" value={g.groundedness_mean.toFixed(2)} sub={`pass ${pct(g.groundedness_pass_rate)}`} />
          <Card label="abstention F1" value={pct(g.abstention.f1)} sub={`hallucination on unanswerable ${pct(g.abstention.hallucination_rate_on_unanswerable)}`} />
        </div>
      )}
      <section>
        <h2 className="mb-2 font-semibold">Retrieval ablation</h2>
        <table className="w-full text-sm">
          <thead className="muted text-left text-xs">
            <tr>
              <th>config</th>
              <th>R@5</th>
              <th>R@10</th>
              <th>R@30</th>
              <th>MRR</th>
              <th>n</th>
            </tr>
          </thead>
          <tbody>
            {r.retrieval.map((row) => (
              <tr key={row.config} className="border-t border-zinc-800">
                <td className="py-1 font-mono">{row.config}</td>
                <td>{pct(row.recall_at_5)}</td>
                <td>{pct(row.recall_at_10)}</td>
                <td>{pct(row.recall_at_30)}</td>
                <td>{row.mrr.toFixed(3)}</td>
                <td>{row.n}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
      {g && (
        <section>
          <h2 className="mb-2 font-semibold">By question type</h2>
          <table className="w-full text-sm">
            <thead className="muted text-left text-xs">
              <tr>
                <th>type</th>
                <th>n</th>
                <th>groundedness</th>
                <th>faithfulness</th>
                <th>relevance</th>
                <th>abstention rate</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(g.by_type).map(([t, m]) => (
                <tr key={t} className="border-t border-zinc-800">
                  <td className="py-1 font-mono">{t}</td>
                  <td>{m.n}</td>
                  <td>{m.groundedness_mean.toFixed(2)}</td>
                  <td>{m.faithfulness_mean ? m.faithfulness_mean.toFixed(2) : '–'}</td>
                  <td>{m.relevance_mean ? m.relevance_mean.toFixed(2) : '–'}</td>
                  <td>{pct(m.abstention_rate)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}
      {r.models && r.models.length > 0 && (
        <section>
          <h2 className="mb-2 font-semibold">Model comparison</h2>
          <table className="w-full text-sm">
            <thead className="muted text-left text-xs">
              <tr>
                <th>model</th>
                <th>faithfulness</th>
                <th>relevance</th>
                <th>groundedness</th>
                <th>abstention F1</th>
              </tr>
            </thead>
            <tbody>
              {r.models.map((m) => (
                <tr key={m.model} className="border-t border-zinc-800">
                  <td className="py-1 font-mono">{m.model}</td>
                  <td>{m.faithfulness_mean.toFixed(2)}</td>
                  <td>{m.relevance_mean.toFixed(2)}</td>
                  <td>{m.groundedness_mean.toFixed(2)}</td>
                  <td>{pct(m.abstention_f1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}
      <section className="card text-sm">
        <h2 className="mb-1 font-semibold">Judge validation</h2>
        {r.judge_agreement ? (
          <p>
            {r.judge_agreement.n} hand-labeled examples: agreement {pct(r.judge_agreement.percent_agreement)}, Cohen's κ {r.judge_agreement.cohen_kappa.toFixed(2)} —{' '}
            {r.judge_agreement.trusted ? <span className="text-emerald-400">judge trusted (≥ 80%)</span> : <span className="text-amber-400">below the 80% bar; judge prompts still being iterated</span>}
          </p>
        ) : (
          <p className="muted">Not yet run — the judge is validated against 30 owner-labeled examples before generation metrics are trusted.</p>
        )}
      </section>
    </div>
  )
}
