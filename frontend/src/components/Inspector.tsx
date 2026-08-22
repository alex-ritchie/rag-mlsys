import { useEffect, useState } from 'react'
import { getChunk, type Chunk, type Citation } from '../lib/api'

/** Retrieval inspector (spec §5.10 #2): reranked top-K with scores; expand to the full chunk text. */
export function Inspector({ citations, active, onHover }: { citations: Citation[]; active: number | null; onHover: (n: number | null) => void }) {
  const [open, setOpen] = useState<number | null>(null)
  const [full, setFull] = useState<Record<number, Chunk>>({})
  useEffect(() => {
    if (open !== null && !full[open]) getChunk(open).then((c) => setFull((f) => ({ ...f, [open]: c }))).catch(() => {})
  }, [open, full])
  if (!citations.length) return <div className="muted text-sm">Sources appear here as soon as retrieval finishes — before the first token.</div>
  return (
    <div className="space-y-2">
      <div className="muted text-xs">
        reranked top-{citations.length} · hover a <span className="chip">n</span> in the answer to highlight
      </div>
      {citations.map((c) => (
        <div
          key={c.chunk_id}
          className={`card cursor-pointer p-3 transition ${active === c.n ? 'ring-2 ring-indigo-500' : ''}`}
          onMouseEnter={() => onHover(c.n)}
          onMouseLeave={() => onHover(null)}
          onClick={() => setOpen(open === c.chunk_id ? null : c.chunk_id)}
        >
          <div className="flex items-start justify-between gap-2">
            <div className="text-xs font-medium">
              <span className="chip mr-1">{c.n}</span>
              {c.heading_path}
            </div>
            <div className="muted shrink-0 text-right font-mono text-[11px]">
              {c.rerank_score !== null && <div>rerank {c.rerank_score.toFixed(2)}</div>}
              <div>rrf {c.fusion_score.toFixed(4)}</div>
              <div>#{c.chunk_id}</div>
            </div>
          </div>
          <div className="muted mt-1 text-xs">{open === c.chunk_id && full[c.chunk_id] ? null : c.text_preview}</div>
          {open === c.chunk_id && full[c.chunk_id] && (
            <pre className="mt-2 max-h-96 overflow-auto whitespace-pre-wrap rounded bg-zinc-950/60 p-2 font-mono text-[11px] leading-snug">
              {full[c.chunk_id].text}
            </pre>
          )}
        </div>
      ))}
    </div>
  )
}
