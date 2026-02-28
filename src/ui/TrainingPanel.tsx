import { useState, useRef, useCallback, useEffect } from 'react'
import { Trainer, type TrainingMetrics } from '../training/trainer'

export function TrainingPanel() {
  const [status, setStatus] = useState<'idle' | 'initializing' | 'training' | 'stopped'>('idle')
  const [metrics, setMetrics] = useState<TrainingMetrics | null>(null)
  const [lossHistory, setLossHistory] = useState<number[]>([])
  const [sampleText, setSampleText] = useState<string>('')
  const [error, setError] = useState<string | null>(null)
  const trainerRef = useRef<Trainer | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)

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
          setLossHistory((prev) => [...prev.slice(-199), m.loss])
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

  // Draw loss chart on canvas
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || lossHistory.length < 2) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const dpr = window.devicePixelRatio || 1
    const rect = canvas.getBoundingClientRect()
    canvas.width = rect.width * dpr
    canvas.height = rect.height * dpr
    ctx.scale(dpr, dpr)
    const W = rect.width
    const H = rect.height

    // Clear
    ctx.clearRect(0, 0, W, H)

    // Compute Y range with padding
    const minLoss = Math.min(...lossHistory)
    const maxLoss = Math.max(...lossHistory)
    const yPad = Math.max((maxLoss - minLoss) * 0.1, 0.01)
    const yMin = minLoss - yPad
    const yMax = maxLoss + yPad

    const pad = { top: 8, right: 8, bottom: 20, left: 40 }
    const plotW = W - pad.left - pad.right
    const plotH = H - pad.top - pad.bottom

    const toX = (i: number) => pad.left + (i / (lossHistory.length - 1)) * plotW
    const toY = (v: number) => pad.top + (1 - (v - yMin) / (yMax - yMin)) * plotH

    // Grid lines and Y labels
    ctx.strokeStyle = '#333'
    ctx.lineWidth = 0.5
    ctx.fillStyle = '#666'
    ctx.font = '10px monospace'
    ctx.textAlign = 'right'
    const nTicks = 4
    for (let i = 0; i <= nTicks; i++) {
      const v = yMin + (i / nTicks) * (yMax - yMin)
      const y = toY(v)
      ctx.beginPath()
      ctx.moveTo(pad.left, y)
      ctx.lineTo(W - pad.right, y)
      ctx.stroke()
      ctx.fillText(v.toFixed(2), pad.left - 4, y + 3)
    }

    // X axis label
    ctx.textAlign = 'center'
    ctx.fillStyle = '#555'
    ctx.fillText(`${lossHistory.length} steps`, W / 2, H - 2)

    // Gradient fill
    const gradient = ctx.createLinearGradient(0, pad.top, 0, pad.top + plotH)
    gradient.addColorStop(0, 'rgba(76, 175, 80, 0.25)')
    gradient.addColorStop(1, 'rgba(76, 175, 80, 0.02)')

    ctx.beginPath()
    ctx.moveTo(toX(0), toY(lossHistory[0]))
    for (let i = 1; i < lossHistory.length; i++) {
      ctx.lineTo(toX(i), toY(lossHistory[i]))
    }
    ctx.lineTo(toX(lossHistory.length - 1), pad.top + plotH)
    ctx.lineTo(toX(0), pad.top + plotH)
    ctx.closePath()
    ctx.fillStyle = gradient
    ctx.fill()

    // Line
    ctx.beginPath()
    ctx.moveTo(toX(0), toY(lossHistory[0]))
    for (let i = 1; i < lossHistory.length; i++) {
      ctx.lineTo(toX(i), toY(lossHistory[i]))
    }
    ctx.strokeStyle = '#4caf50'
    ctx.lineWidth = 1.5
    ctx.stroke()

    // Current value dot
    const lastIdx = lossHistory.length - 1
    ctx.beginPath()
    ctx.arc(toX(lastIdx), toY(lossHistory[lastIdx]), 3, 0, Math.PI * 2)
    ctx.fillStyle = '#4caf50'
    ctx.fill()
  }, [lossHistory])

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
        <div className="loss-chart-container">
          <h3>Loss</h3>
          <canvas ref={canvasRef} className="loss-canvas" />
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
