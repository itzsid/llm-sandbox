import { useState, useRef, useCallback } from 'react'
import { Trainer, type TrainingMetrics } from '../training/trainer'

export function TrainingPanel() {
  const [status, setStatus] = useState<'idle' | 'initializing' | 'training' | 'stopped'>('idle')
  const [metrics, setMetrics] = useState<TrainingMetrics | null>(null)
  const [lossHistory, setLossHistory] = useState<number[]>([])
  const [sampleText, setSampleText] = useState<string>('')
  const [error, setError] = useState<string | null>(null)
  const trainerRef = useRef<Trainer | null>(null)

  const handleStart = useCallback(async () => {
    try {
      setError(null)
      setStatus('initializing')

      const trainer = new Trainer()
      trainerRef.current = trainer
      await trainer.init()

      setStatus('training')
      await trainer.train(
        (m) => {
          setMetrics(m)
          setLossHistory((prev) => [...prev.slice(-19), m.loss])
        },
        (text) => {
          setSampleText(text)
        },
      )
      setStatus('stopped')
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setStatus('idle')
    }
  }, [])

  const handleStop = useCallback(() => {
    trainerRef.current?.stop()
    setStatus('stopped')
  }, [])

  return (
    <div className="training-panel">
      <div className="controls">
        {(status === 'idle' || status === 'stopped') && (
          <button onClick={handleStart} className="btn-start">
            Start Training
          </button>
        )}
        {status === 'training' && (
          <button onClick={handleStop} className="btn-stop">
            Stop
          </button>
        )}
        {status === 'initializing' && (
          <button disabled className="btn-disabled">
            Initializing...
          </button>
        )}
      </div>

      {error && <div className="error-msg">{error}</div>}

      {metrics && (
        <div className="metrics">
          <div className="metric">
            <span className="metric-label">Step</span>
            <span className="metric-value">{metrics.step}</span>
          </div>
          <div className="metric">
            <span className="metric-label">Loss</span>
            <span className="metric-value">{metrics.loss.toFixed(4)}</span>
          </div>
          <div className="metric">
            <span className="metric-label">Tokens/s</span>
            <span className="metric-value">{metrics.tokensPerSec.toFixed(0)}</span>
          </div>
        </div>
      )}

      {lossHistory.length > 0 && (
        <div className="loss-history">
          <h3>Loss History</h3>
          <div className="loss-list">
            {lossHistory.map((loss, i) => (
              <div key={i} className="loss-bar-row">
                <span className="loss-step">{metrics ? metrics.step - lossHistory.length + i + 1 : i + 1}</span>
                <div className="loss-bar-bg">
                  <div
                    className="loss-bar"
                    style={{ width: `${Math.min(100, (loss / Math.max(...lossHistory)) * 100)}%` }}
                  />
                </div>
                <span className="loss-value">{loss.toFixed(3)}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {sampleText && (
        <div className="sample-output">
          <h3>Generated Sample</h3>
          <pre className="sample-text">{sampleText}</pre>
        </div>
      )}
    </div>
  )
}
