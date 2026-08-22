import { API_BASE } from './config'

export interface Citation {
  n: number
  chunk_id: number
  heading_path: string
  rerank_score: number | null
  fusion_score: number
  text_preview: string
}
export interface LatencyBreakdown {
  embed_ms: number
  retrieve_ms: number
  rerank_ms: number
  ttft_ms: number
  generate_ms: number
  total_ms: number
}
export interface DoneEvent {
  usage: { prompt_tokens: number; completion_tokens: number; total_tokens: number }
  latency_breakdown: LatencyBreakdown
  model: string
  prompt_version: string
  abstained: boolean
  query_log_id: number | null
  demo_cost_usd?: number
}
export interface Chunk {
  id: number
  volume: number
  chapter_num: number
  chapter_title: string
  section_path: string[]
  heading_path: string
  source_file: string
  token_count: number
  commit_sha: string
  content_hash: string
  text: string
  oversize: boolean
}
export interface Coverage {
  commit_sha: string | null
  window_days: number
  total_chunks: number
  volumes: {
    volume: number
    chunks: number
    chapters: {
      chapter_num: number
      title: string
      chunks: number
      tokens: number
      answers: number
      citations: number
      sections: { title: string; chunks: number }[]
    }[]
  }[]
}
export interface RuntimeConfig {
  profile: string
  model: string
  index_commit_sha: string | null
  chunks: number
  prompt_version: string
}

export class RateLimitError extends Error {
  constructor(
    public reason: string,
    message: string,
  ) {
    super(message)
  }
}

export type AskHandlers = {
  onCitations: (c: Citation[]) => void
  onToken: (t: string) => void
  onDone: (d: DoneEvent) => void
  onError: (msg: string) => void
}

/** SSE over fetch + ReadableStream (POST body needed, so no EventSource). */
export async function ask(question: string, h: AskHandlers, opts: { top_k?: number; mode?: 'hybrid' | 'dense' } = {}, signal?: AbortSignal) {
  const r = await fetch(`${API_BASE}/api/ask`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ question, ...opts }),
    signal,
  })
  if (r.status === 429) {
    const body = await r.json()
    throw new RateLimitError(body.error ?? 'rate_limit', body.message ?? 'Rate limited')
  }
  if (!r.ok || !r.body) throw new Error(`HTTP ${r.status}`)
  const reader = r.body.getReader()
  const dec = new TextDecoder()
  let buf = ''
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    buf += dec.decode(value, { stream: true })
    let idx: number
    while ((idx = buf.search(/\r?\n\r?\n/)) >= 0) {
      const block = buf.slice(0, idx)
      buf = buf.slice(idx).replace(/^\r?\n\r?\n/, '')
      let ev = 'message'
      const data: string[] = []
      for (const line of block.split(/\r?\n/)) {
        if (line.startsWith('event:')) ev = line.slice(6).trim()
        else if (line.startsWith('data:')) data.push(line.slice(5).trimStart())
      }
      if (!data.length) continue
      const payload = JSON.parse(data.join('\n'))
      if (ev === 'citations') h.onCitations(payload as Citation[])
      else if (ev === 'token') h.onToken((payload as { text: string }).text)
      else if (ev === 'done') h.onDone(payload as DoneEvent)
      else if (ev === 'error') h.onError((payload as { message: string }).message)
    }
  }
}

export const getChunk = (id: number) => fetch(`${API_BASE}/api/chunks/${id}`).then((r) => r.json() as Promise<Chunk>)
export const getCoverage = () => fetch(`${API_BASE}/api/coverage`).then((r) => r.json() as Promise<Coverage>)
export const getConfig = () => fetch(`${API_BASE}/api/config`).then((r) => r.json() as Promise<RuntimeConfig>)
