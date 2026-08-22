import { useEffect, useState } from 'react'
import { getCoverage, type Coverage } from '../lib/api'

function label(n: number) {
  if (n >= 200) return 'Glossary'
  if (n >= 100) return `Appendix ${String.fromCharCode(65 + n - 100)}`
  return `Ch ${n}`
}

export default function CoveragePage() {
  const [cov, setCov] = useState<Coverage | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [open, setOpen] = useState<string | null>(null)
  useEffect(() => {
    getCoverage().then(setCov).catch((e) => setErr(String(e)))
  }, [])
  if (err) return <div className="text-red-400">coverage unavailable: {err}</div>
  if (!cov) return <div className="muted">loading…</div>
  const maxChunks = Math.max(...cov.volumes.flatMap((v) => v.chapters.map((c) => c.chunks)))
  return (
    <div className="space-y-6">
      <div className="muted text-sm">
        {cov.total_chunks} chunks · index built from commit <code className="font-mono">{cov.commit_sha?.slice(0, 10)}</code> · bars = chunk count, dots = answers in the last{' '}
        {cov.window_days} days that cited the chapter
      </div>
      {cov.volumes.map((v) => (
        <section key={v.volume}>
          <h2 className="mb-2 font-semibold">
            Volume {v.volume} <span className="muted text-xs">({v.chunks} chunks)</span>
          </h2>
          <div className="space-y-1">
            {v.chapters.map((c) => {
              const key = `${v.volume}-${c.chapter_num}`
              return (
                <div key={key} className="card cursor-pointer p-2" onClick={() => setOpen(open === key ? null : key)}>
                  <div className="flex items-center gap-3 text-sm">
                    <span className="muted w-24 shrink-0 font-mono text-xs">{label(c.chapter_num)}</span>
                    <span className="w-56 shrink-0 truncate">{c.title}</span>
                    <div className="h-2 flex-1 rounded bg-zinc-800">
                      <div className="h-2 rounded bg-indigo-500" style={{ width: `${(100 * c.chunks) / maxChunks}%` }} />
                    </div>
                    <span className="muted w-16 shrink-0 text-right font-mono text-xs">{c.chunks}</span>
                    <span className="w-20 shrink-0 text-right font-mono text-xs" title={`${c.citations} citations`}>
                      {c.answers > 0 ? `● ${c.answers}` : <span className="muted">–</span>}
                    </span>
                  </div>
                  {open === key && (
                    <ul className="muted mt-2 grid gap-x-4 pl-24 text-xs sm:grid-cols-2">
                      {c.sections.map((s) => (
                        <li key={s.title}>
                          {s.title || '(chapter intro)'} <span className="font-mono">{s.chunks}</span>
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              )
            })}
          </div>
        </section>
      ))}
    </div>
  )
}
