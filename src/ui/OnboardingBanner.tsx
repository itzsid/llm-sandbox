import { useState } from 'react'

const STORAGE_KEY = 'llm-sandbox-onboarding-dismissed'

interface OnboardingBannerProps {
  onQuickStart: () => void
}

export function OnboardingBanner({ onQuickStart }: OnboardingBannerProps) {
  const [dismissed, setDismissed] = useState(() => {
    try {
      return localStorage.getItem(STORAGE_KEY) === 'true'
    } catch {
      return false
    }
  })

  if (dismissed) return null

  const handleDismiss = () => {
    setDismissed(true)
    try {
      localStorage.setItem(STORAGE_KEY, 'true')
    } catch {
      // ignore
    }
  }

  const handleQuickStart = () => {
    handleDismiss()
    onQuickStart()
  }

  return (
    <div style={styles.banner}>
      <div style={styles.content}>
        <div style={styles.title}>Welcome to LLM Sandbox</div>
        <div style={styles.description}>
          Train small language models directly in your browser using WebGPU.
          Define architectures, visualize training, and generate text — all client-side.
        </div>
        <div style={styles.actions}>
          <button className="btn btn-primary" onClick={handleQuickStart}>
            Quick Start: Train Nano on Shakespeare
          </button>
          <button className="btn btn-outline" onClick={handleDismiss}>
            Dismiss
          </button>
        </div>
      </div>
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  banner: {
    background: 'linear-gradient(135deg, var(--bg-surface) 0%, var(--bg-elevated) 100%)',
    border: '1px solid var(--border-active)',
    borderRadius: 'var(--radius-lg)',
    padding: '1.25rem',
    marginBottom: '1.5rem',
  },
  content: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.75rem',
  },
  title: {
    fontSize: '1.1rem',
    fontWeight: 700,
    color: 'var(--text-1)',
    fontFamily: 'var(--font-display)',
  },
  description: {
    color: 'var(--text-2)',
    fontSize: '0.9rem',
    lineHeight: 1.5,
  },
  actions: {
    display: 'flex',
    gap: '0.75rem',
    flexWrap: 'wrap',
  },
}
