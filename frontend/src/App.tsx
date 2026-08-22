import { NavLink, Route, Routes } from 'react-router-dom'
import { useTheme } from './lib/theme'
import { PROFILE, BOOK } from './lib/config'
import ChatPage from './pages/ChatPage'
import CoveragePage from './pages/CoveragePage'
import EvalPage from './pages/EvalPage'
import AboutPage from './pages/AboutPage'

const tabs = [
  ['/', 'Chat'],
  ['/coverage', 'Coverage'],
  ['/eval', 'Eval'],
  ['/about', 'About'],
] as const

export default function App() {
  const { dark, toggle } = useTheme()
  return (
    <div className="mx-auto flex min-h-screen max-w-7xl flex-col px-4">
      <header className="flex items-center justify-between border-b border-zinc-800 py-3">
        <div className="flex items-baseline gap-3">
          <h1 className="text-lg font-semibold">MLSysBook Companion</h1>
          <span className="muted text-xs">
            RAG over <em>{BOOK.title}</em>
            {PROFILE === 'demo' && <span className="ml-2 rounded bg-amber-500/20 px-1.5 py-0.5 text-amber-300">hosted demo</span>}
          </span>
        </div>
        <nav className="flex items-center gap-1 text-sm">
          {tabs.map(([to, label]) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) => `rounded px-2 py-1 ${isActive ? 'bg-zinc-800 text-white' : 'muted hover:text-zinc-200'}`}
            >
              {label}
            </NavLink>
          ))}
          <button onClick={toggle} className="muted ml-2 rounded px-2 py-1 hover:text-zinc-200" aria-label="toggle theme">
            {dark ? '☾' : '☀'}
          </button>
        </nav>
      </header>
      <main className="flex-1 py-4">
        <Routes>
          <Route path="/" element={<ChatPage />} />
          <Route path="/coverage" element={<CoveragePage />} />
          <Route path="/eval" element={<EvalPage />} />
          <Route path="/about" element={<AboutPage />} />
        </Routes>
      </main>
      <footer className="muted border-t border-zinc-800 py-3 text-xs">
        Content from <a className="underline" href={BOOK.url}>{BOOK.title}</a> by {BOOK.author}, licensed{' '}
        <a className="underline" href={BOOK.licenseUrl}>{BOOK.license}</a>. Answers cite the book; check the sources.
        {PROFILE === 'demo' && (
          <>
            {' '}
            This hosted demo is an accessibility layer over a cheap API model; the local k8s/vLLM stack is the actual project —
            see <NavLink to="/about" className="underline">About</NavLink>.
          </>
        )}
      </footer>
    </div>
  )
}
