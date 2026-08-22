import { useRef, useState } from 'react'
import { ask, RateLimitError, type Citation, type DoneEvent } from '../lib/api'
import { AnswerText } from '../components/Markdown'
import { Inspector } from '../components/Inspector'
import { LatencyFooter } from '../components/LatencyFooter'
import { REPO_URL, PROFILE } from '../lib/config'

interface Turn {
  id: number
  question: string
  answer: string
  citations: Citation[]
  done?: DoneEvent
  error?: string
  limited?: string
}

const EXAMPLES = [
  'What is the difference between Software 1.0 and Software 2.0?',
  'How does INT8 quantization change memory and accuracy?',
  'Why is the KV cache a bottleneck for LLM inference?',
  'What does the roofline model tell you about a kernel?',
]

export default function ChatPage() {
  const [turns, setTurns] = useState<Turn[]>([])
  const [q, setQ] = useState('')
  const [busy, setBusy] = useState(false)
  const [selected, setSelected] = useState<number | null>(null)
  const [hover, setHover] = useState<number | null>(null)
  const [mode, setMode] = useState<'hybrid' | 'dense'>('hybrid')
  const abort = useRef<AbortController | null>(null)

  const current = turns.find((t) => t.id === selected) ?? turns[turns.length - 1]

  async function submit(question: string) {
    if (!question.trim() || busy) return
    const id = Date.now()
    setTurns((ts) => [...ts, { id, question, answer: '', citations: [] }])
    setSelected(id)
    setQ('')
    setBusy(true)
    abort.current = new AbortController()
    const patch = (p: Partial<Turn> | ((t: Turn) => Partial<Turn>)) =>
      setTurns((ts) => ts.map((t) => (t.id === id ? { ...t, ...(typeof p === 'function' ? p(t) : p) } : t)))
    try {
      await ask(
        question,
        {
          onCitations: (citations) => patch({ citations }),
          onToken: (tok) => patch((t) => ({ answer: t.answer + tok })),
          onDone: (done) => patch({ done }),
          onError: (error) => patch({ error }),
        },
        { mode },
        abort.current.signal,
      )
    } catch (e) {
      if (e instanceof RateLimitError) patch({ limited: e.message })
      else patch({ error: (e as Error).message })
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="grid gap-4 lg:grid-cols-[1fr_380px]">
      <section className="flex min-h-[70vh] flex-col">
        <div className="flex-1 space-y-4">
          {turns.length === 0 && (
            <div className="card">
              <p className="mb-2 text-sm">Ask anything covered by the two volumes. Answers are grounded in retrieved passages and cite them inline.</p>
              <div className="flex flex-wrap gap-2">
                {EXAMPLES.map((e) => (
                  <button key={e} className="rounded border border-zinc-700 px-2 py-1 text-left text-xs hover:border-indigo-500" onClick={() => submit(e)}>
                    {e}
                  </button>
                ))}
              </div>
            </div>
          )}
          {turns.map((t) => (
            <div key={t.id} className={`card ${t.id === current?.id ? '' : 'opacity-80'}`} onClick={() => setSelected(t.id)}>
              <div className="mb-2 text-sm font-medium text-indigo-300">{t.question}</div>
              {t.limited ? (
                <div className="rounded border border-amber-500/40 bg-amber-500/10 p-2 text-sm">
                  {t.limited}{' '}
                  <a className="underline" href={`${REPO_URL}#run-it-yourself`}>
                    Run it yourself →
                  </a>
                </div>
              ) : t.error ? (
                <div className="text-sm text-red-400">error: {t.error}</div>
              ) : (
                <AnswerText text={t.answer || (busy && t.id === selected ? '…' : '')} onCite={setHover} active={hover} />
              )}
              {t.done && <LatencyFooter done={t.done} />}
            </div>
          ))}
        </div>
        <form
          className="sticky bottom-0 mt-4 flex gap-2 bg-zinc-950/80 py-2 backdrop-blur"
          onSubmit={(e) => {
            e.preventDefault()
            submit(q)
          }}
        >
          <input className="input" value={q} onChange={(e) => setQ(e.target.value)} placeholder="Ask about the book…" disabled={busy} />
          <select className="input w-auto" value={mode} onChange={(e) => setMode(e.target.value as 'hybrid' | 'dense')} title="retrieval mode">
            <option value="hybrid">hybrid</option>
            <option value="dense">dense</option>
          </select>
          {busy ? (
            <button type="button" className="btn" onClick={() => abort.current?.abort()}>
              stop
            </button>
          ) : (
            <button className="btn" type="submit" disabled={!q.trim()}>
              ask
            </button>
          )}
        </form>
        {PROFILE === 'demo' && <p className="muted mt-1 text-[11px]">Demo: 10 questions/day per IP, generation by Claude Haiku. The local stack serves Qwen3.8-27B on a single 3090 Ti.</p>}
      </section>
      <aside className="lg:sticky lg:top-4 lg:max-h-[90vh] lg:overflow-auto">
        <h2 className="mb-2 text-sm font-semibold">Retrieval inspector</h2>
        <Inspector citations={current?.citations ?? []} active={hover} onHover={setHover} />
      </aside>
    </div>
  )
}
