import type { DoneEvent } from '../lib/api'

export function LatencyFooter({ done }: { done: DoneEvent }) {
  const l = done.latency_breakdown
  const cells: [string, number][] = [
    ['embed', l.embed_ms],
    ['retrieve', l.retrieve_ms],
    ['rerank', l.rerank_ms],
    ['TTFT', l.ttft_ms],
    ['generate', l.generate_ms],
    ['total', l.total_ms],
  ]
  return (
    <div className="muted mt-2 flex flex-wrap gap-x-3 font-mono text-[11px]">
      {cells.map(([k, v]) => (
        <span key={k}>
          {k} <span className="text-zinc-300">{v.toFixed(0)}ms</span>
        </span>
      ))}
      <span>
        tokens <span className="text-zinc-300">{done.usage.prompt_tokens}→{done.usage.completion_tokens}</span>
      </span>
      <span>
        {done.model} · {done.prompt_version}
      </span>
      {done.abstained && <span className="text-amber-400">abstained</span>}
      {done.demo_cost_usd !== undefined && <span>${done.demo_cost_usd.toFixed(4)}</span>}
    </div>
  )
}
