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
      await trainer.generateStreaming(
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

      // Get final text
      const finalText = await trainer.generateSample(maxTokens, temperature, prompt || undefined)
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
          className="btn btn-blue"
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
              <span style={{ color: '#666' }}>{output.slice(0, promptLen)}</span>
            )}
            {/* Generated continuation in bright color */}
            <span className="playground-token" style={{ color: '#e0e0e0' }}>
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
    background: '#1a1a1a',
    border: '1px solid #333',
    borderRadius: '6px',
    marginTop: '1.5rem',
  },
  heading: {
    fontSize: '0.85rem',
    color: '#888',
    margin: '0 0 0.75rem',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
  },
  promptArea: {
    marginBottom: '0.75rem',
  },
  promptInput: {
    width: '100%',
    padding: '0.75rem',
    background: '#222',
    border: '1px solid #444',
    borderRadius: '4px',
    color: '#e0e0e0',
    fontSize: '0.9rem',
    fontFamily: "'SF Mono', 'Fira Code', monospace",
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
    color: '#aaa',
  },
  slider: {
    width: '100%',
    accentColor: '#2196f3',
  },
  outputArea: {
    marginTop: '0.5rem',
  },
  outputText: {
    background: '#111',
    border: '1px solid #333',
    borderRadius: '4px',
    padding: '1rem',
    fontSize: '0.9rem',
    fontFamily: "'SF Mono', 'Fira Code', monospace",
    whiteSpace: 'pre-wrap',
    wordBreak: 'break-all',
    maxHeight: '300px',
    overflowY: 'auto',
    margin: 0,
    lineHeight: 1.6,
  },
  cursor: {
    color: '#4caf50',
    animation: 'blink 1s step-end infinite',
  },
  hint: {
    color: '#666',
    fontSize: '0.85rem',
    fontStyle: 'italic',
    textAlign: 'center',
    padding: '2rem 0',
  },
  history: {
    marginTop: '1rem',
    borderTop: '1px solid #333',
    paddingTop: '0.75rem',
  },
  historyHeading: {
    fontSize: '0.75rem',
    color: '#888',
    margin: '0 0 0.5rem',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
  },
  historyEntry: {
    padding: '0.5rem',
    background: '#222',
    borderRadius: '4px',
    marginBottom: '0.5rem',
    border: '1px solid #333',
  },
  historyPrompt: {
    color: '#999',
    fontSize: '0.75rem',
    marginBottom: '0.25rem',
  },
  historyCompletion: {
    color: '#ccc',
    fontSize: '0.8rem',
    fontFamily: "'SF Mono', 'Fira Code', monospace",
  },
}
