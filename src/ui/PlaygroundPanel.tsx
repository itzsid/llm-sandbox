import { useState, useCallback, useRef } from 'react'
import type { Trainer } from '../training/trainer'

interface PlaygroundPanelProps {
  trainer: Trainer | null
  isTraining: boolean
}

interface HistoryEntry {
  prompt: string
  completion: string
  timestamp: number
}

export function PlaygroundPanel({ trainer, isTraining }: PlaygroundPanelProps) {
  const [prompt, setPrompt] = useState('')
  const [output, setOutput] = useState('')
  const [promptLen, setPromptLen] = useState(0)
  const [generating, setGenerating] = useState(false)
  const [temperature, setTemperature] = useState(0.8)
  const [maxTokens, setMaxTokens] = useState(200)
  const [history, setHistory] = useState<HistoryEntry[]>([])
  const outputRef = useRef<HTMLPreElement>(null)

  const canGenerate = trainer?.params && !isTraining && !generating

  const handleGenerate = useCallback(async () => {
    if (!trainer?.params) return
    setGenerating(true)
    setOutput('')
    setPromptLen(prompt.length)

    try {
      const finalText = await trainer.generateStreaming(
        maxTokens,
        temperature,
        (text) => {
          setOutput(text)
          if (outputRef.current) {
            outputRef.current.scrollTop = outputRef.current.scrollHeight
          }
        },
        prompt || undefined,
      )

      setOutput(finalText)

      // Add to history
      setHistory((prev) => [
        { prompt: prompt || '(random)', completion: finalText, timestamp: Date.now() },
        ...prev.slice(0, 9), // keep last 10
      ])
    } catch (e) {
      setOutput(`Error: ${e instanceof Error ? e.message : String(e)}`)
    } finally {
      setGenerating(false)
    }
  }, [trainer, prompt, temperature, maxTokens])

  return (
    <div style={styles.container}>
      <h3 style={styles.heading}>Playground</h3>

      <div style={styles.promptArea}>
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Enter a prompt (or leave empty for random start)..."
          style={styles.promptInput}
          rows={3}
        />
      </div>

      <div style={styles.controls}>
        <div style={styles.sliderGroup}>
          <label style={styles.sliderLabel}>Temperature: {temperature.toFixed(1)}</label>
          <input
            type="range"
            min="0.1"
            max="2.0"
            step="0.1"
            value={temperature}
            onChange={(e) => setTemperature(parseFloat(e.target.value))}
            style={styles.slider}
          />
        </div>
        <div style={styles.sliderGroup}>
          <label style={styles.sliderLabel}>Max Tokens: {maxTokens}</label>
          <input
            type="range"
            min="50"
            max="500"
            step="50"
            value={maxTokens}
            onChange={(e) => setMaxTokens(parseInt(e.target.value))}
            style={styles.slider}
          />
        </div>
        <button
          className="btn btn-primary"
          onClick={handleGenerate}
          disabled={!canGenerate}
        >
          {generating ? 'Generating...' : 'Generate'}
        </button>
      </div>

      {output && (
        <div style={styles.outputArea}>
          <pre ref={outputRef} style={styles.outputText}>
            {/* Prompt text in dim color */}
            {promptLen > 0 && (
              <span style={{ color: 'var(--text-3)' }}>{output.slice(0, promptLen)}</span>
            )}
            {/* Generated continuation in bright color */}
            <span className="playground-token" style={{ color: 'var(--text-1)' }}>
              {output.slice(promptLen)}
            </span>
            {generating && <span style={styles.cursor}>|</span>}
          </pre>
        </div>
      )}

      {!trainer?.params && (
        <div style={styles.hint}>
          Train a model first, then generate text here.
        </div>
      )}

      {history.length > 0 && (
        <div style={styles.history}>
          <h4 style={styles.historyHeading}>History</h4>
          {history.map((entry, i) => (
            <div key={i} style={styles.historyEntry}>
              <div style={styles.historyPrompt}>{entry.prompt}</div>
              <div style={styles.historyCompletion}>
                {entry.completion.slice(0, 100)}
                {entry.completion.length > 100 ? '...' : ''}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    padding: '1rem',
    background: 'var(--bg-surface)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-md)',
    marginTop: '1.5rem',
  },
  heading: {
    fontSize: '0.85rem',
    color: 'var(--text-3)',
    margin: '0 0 0.75rem',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
    fontFamily: 'var(--font-body)',
  },
  promptArea: {
    marginBottom: '0.75rem',
  },
  promptInput: {
    width: '100%',
    padding: '0.75rem',
    background: 'var(--bg-elevated)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-sm)',
    color: 'var(--text-1)',
    fontSize: '0.9rem',
    fontFamily: 'var(--font-mono)',
    outline: 'none',
    resize: 'vertical',
    boxSizing: 'border-box',
  },
  controls: {
    display: 'flex',
    gap: '1rem',
    alignItems: 'flex-end',
    flexWrap: 'wrap',
    marginBottom: '0.75rem',
  },
  sliderGroup: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.25rem',
    flex: 1,
    minWidth: '120px',
  },
  sliderLabel: {
    fontSize: '0.75rem',
    color: 'var(--text-2)',
    fontFamily: 'var(--font-body)',
  },
  slider: {
    width: '100%',
    accentColor: 'var(--amber)',
  },
  outputArea: {
    marginTop: '0.5rem',
  },
  outputText: {
    background: 'var(--bg-base)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-sm)',
    padding: '1rem',
    fontSize: '0.9rem',
    fontFamily: 'var(--font-mono)',
    whiteSpace: 'pre-wrap',
    wordBreak: 'break-all',
    maxHeight: '300px',
    overflowY: 'auto',
    margin: 0,
    lineHeight: 1.6,
  },
  cursor: {
    color: 'var(--amber)',
    animation: 'blink 1s step-end infinite',
  },
  hint: {
    color: 'var(--text-3)',
    fontSize: '0.85rem',
    fontStyle: 'italic',
    textAlign: 'center',
    padding: '2rem 0',
  },
  history: {
    marginTop: '1rem',
    borderTop: '1px solid var(--border)',
    paddingTop: '0.75rem',
  },
  historyHeading: {
    fontSize: '0.75rem',
    color: 'var(--text-3)',
    margin: '0 0 0.5rem',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
    fontFamily: 'var(--font-body)',
  },
  historyEntry: {
    padding: '0.5rem',
    background: 'var(--bg-elevated)',
    borderRadius: 'var(--radius-sm)',
    marginBottom: '0.5rem',
    border: '1px solid var(--border)',
  },
  historyPrompt: {
    color: 'var(--text-2)',
    fontSize: '0.75rem',
    marginBottom: '0.25rem',
  },
  historyCompletion: {
    color: 'var(--text-2)',
    fontSize: '0.8rem',
    fontFamily: 'var(--font-mono)',
  },
}
