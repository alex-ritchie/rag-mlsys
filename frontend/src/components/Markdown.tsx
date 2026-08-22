import type { ReactNode } from 'react'

/** Minimal renderer: paragraphs, **bold**, `code`, and [n] citation chips. No external markdown lib (keeps the bundle tiny). */
export function AnswerText({
  text,
  onCite,
  active,
}: {
  text: string
  onCite?: (n: number | null) => void
  active: number | null
}) {
  const paras = text.split(/\n{2,}/)
  return (
    <div className="space-y-2 leading-relaxed">
      {paras.map((p, i) => (
        <p key={i}>{renderInline(p, onCite, active)}</p>
      ))}
    </div>
  )
}

function renderInline(s: string, onCite: ((n: number | null) => void) | undefined, active: number | null): ReactNode[] {
  const out: ReactNode[] = []
  const re = /(\[(\d{1,2})\])|(\*\*(.+?)\*\*)|(`([^`]+)`)/g
  let last = 0
  let m: RegExpExecArray | null
  let k = 0
  while ((m = re.exec(s))) {
    if (m.index > last) out.push(s.slice(last, m.index))
    if (m[2]) {
      const n = Number(m[2])
      out.push(
        <span
          key={k++}
          className={`chip mx-0.5 ${active === n ? 'active' : ''}`}
          onMouseEnter={() => onCite?.(n)}
          onMouseLeave={() => onCite?.(null)}
          onClick={() => onCite?.(n)}
        >
          {n}
        </span>,
      )
    } else if (m[4]) out.push(<strong key={k++}>{m[4]}</strong>)
    else if (m[6])
      out.push(
        <code key={k++} className="rounded bg-zinc-800 px-1 font-mono text-[0.85em]">
          {m[6]}
        </code>,
      )
    last = m.index + m[0].length
  }
  if (last < s.length) out.push(s.slice(last))
  return out
}
