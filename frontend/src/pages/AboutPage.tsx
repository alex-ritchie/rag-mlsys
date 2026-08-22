import { useEffect, useState } from 'react'
import { getConfig, type RuntimeConfig } from '../lib/api'
import { BOOK, PROFILE, REPO_URL } from '../lib/config'

export default function AboutPage() {
  const [cfg, setCfg] = useState<RuntimeConfig | null>(null)
  useEffect(() => {
    getConfig().then(setCfg).catch(() => {})
  }, [])
  return (
    <div className="prose-sm max-w-3xl space-y-6">
      <section className="card">
        <h2 className="mb-2 font-semibold">Attribution</h2>
        <p>
          All answers are grounded in <a className="underline" href={BOOK.url}>{BOOK.title}</a> by <strong>{BOOK.author}</strong> (Harvard University), licensed{' '}
          <a className="underline" href={BOOK.licenseUrl}>{BOOK.license}</a>. Source: <a className="underline" href={BOOK.source}>harvard-edge/cs249r_book</a>.
          This project is an independent study companion, not affiliated with the author; it never republishes the book — every user builds their own index from the
          source repository.
        </p>
      </section>
      <section className="card">
        <h2 className="mb-2 font-semibold">Index provenance</h2>
        {cfg ? (
          <ul className="text-sm">
            <li>
              index built from commit <code className="font-mono">{cfg.index_commit_sha}</code> ({cfg.chunks} chunks)
            </li>
            <li>generation model: {cfg.model}</li>
            <li>prompt version: {cfg.prompt_version}</li>
            <li>profile: {cfg.profile}</li>
          </ul>
        ) : (
          <p className="muted text-sm">gateway not reachable</p>
        )}
      </section>
      <section className="card">
        <h2 className="mb-2 font-semibold">{PROFILE === 'demo' ? 'This is the hosted demo, not the system' : 'Local stack'}</h2>
        <p className="text-sm">
          The project is a local ML-systems stack: bge-m3 + Postgres/pgvector hybrid retrieval, bge-reranker-v2-m3, and <strong>Qwen3.8-27B (W4A16) served by vLLM on a
          single RTX 3090 Ti</strong>, deployed on k3s with Prometheus/Grafana monitoring and a nightly groundedness judge. The hosted demo swaps only the generation
          model for Claude Haiku so you can try it without a GPU. Benchmarks, the four-model ablation, and the run-it-yourself guide live in the repo:{' '}
          <a className="underline" href={REPO_URL}>
            {REPO_URL.replace('https://', '')}
          </a>
          .
        </p>
      </section>
    </div>
  )
}
