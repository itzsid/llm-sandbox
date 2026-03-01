import { useRef, useEffect } from 'react'
import { computeLR, type TrainingHyperparams } from '../training/trainer'

interface LRScheduleChartProps {
  hyperparams: TrainingHyperparams
  currentStep: number
  height?: number
}

const NUM_POINTS = 300

export function LRScheduleChart({ hyperparams, currentStep, height = 100 }: LRScheduleChartProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const dpr = window.devicePixelRatio || 1
    const rect = canvas.getBoundingClientRect()
    canvas.width = rect.width * dpr
    canvas.height = rect.height * dpr
    ctx.scale(dpr, dpr)
    const W = rect.width
    const H = rect.height

    ctx.clearRect(0, 0, W, H)

    // Precompute the full schedule curve
    const maxSteps = hyperparams.maxSteps
    const data: number[] = []
    for (let i = 0; i < NUM_POINTS; i++) {
      const step = Math.floor((i / (NUM_POINTS - 1)) * maxSteps)
      data.push(computeLR(step, hyperparams))
    }

    const minVal = Math.min(...data)
    const maxVal = Math.max(...data)
    const yPad = Math.max((maxVal - minVal) * 0.1, 1e-6)
    const yMin = minVal - yPad
    const yMax = maxVal + yPad

    const pad = { top: 8, right: 8, bottom: 20, left: 50 }
    const plotW = W - pad.left - pad.right
    const plotH = H - pad.top - pad.bottom

    const toX = (i: number) => pad.left + (i / (NUM_POINTS - 1)) * plotW
    const toY = (v: number) => pad.top + (1 - (v - yMin) / (yMax - yMin)) * plotH

    // Grid lines and Y labels
    ctx.strokeStyle = 'rgba(255,255,255,0.06)'
    ctx.lineWidth = 0.5
    ctx.fillStyle = '#4B5563'
    ctx.font = "10px 'JetBrains Mono', monospace"
    ctx.textAlign = 'right'
    const nTicks = 4
    for (let i = 0; i <= nTicks; i++) {
      const v = yMin + (i / nTicks) * (yMax - yMin)
      const y = toY(v)
      ctx.beginPath()
      ctx.moveTo(pad.left, y)
      ctx.lineTo(W - pad.right, y)
      ctx.stroke()
      ctx.fillText(v.toExponential(1), pad.left - 4, y + 3)
    }

    // X axis label
    ctx.textAlign = 'center'
    ctx.fillStyle = '#4B5563'
    ctx.fillText(`${maxSteps} steps`, W / 2, H - 2)

    const color = '#F472B6'
    const r = 244, g = 114, b = 182

    // Gradient fill
    const gradient = ctx.createLinearGradient(0, pad.top, 0, pad.top + plotH)
    gradient.addColorStop(0, `rgba(${r}, ${g}, ${b}, 0.25)`)
    gradient.addColorStop(1, `rgba(${r}, ${g}, ${b}, 0.02)`)

    ctx.beginPath()
    ctx.moveTo(toX(0), toY(data[0]))
    for (let i = 1; i < data.length; i++) {
      ctx.lineTo(toX(i), toY(data[i]))
    }
    ctx.lineTo(toX(data.length - 1), pad.top + plotH)
    ctx.lineTo(toX(0), pad.top + plotH)
    ctx.closePath()
    ctx.fillStyle = gradient
    ctx.fill()

    // Line
    ctx.beginPath()
    ctx.moveTo(toX(0), toY(data[0]))
    for (let i = 1; i < data.length; i++) {
      ctx.lineTo(toX(i), toY(data[i]))
    }
    ctx.strokeStyle = color
    ctx.lineWidth = 1.5
    ctx.stroke()

    // Current step marker
    if (currentStep > 0 && currentStep <= maxSteps) {
      const markerFrac = currentStep / maxSteps
      const markerX = pad.left + markerFrac * plotW
      const markerLR = computeLR(currentStep, hyperparams)
      const markerY = toY(markerLR)

      // Vertical line at current step
      ctx.strokeStyle = 'rgba(255,255,255,0.3)'
      ctx.lineWidth = 1
      ctx.setLineDash([3, 3])
      ctx.beginPath()
      ctx.moveTo(markerX, pad.top)
      ctx.lineTo(markerX, pad.top + plotH)
      ctx.stroke()
      ctx.setLineDash([])

      // Dot
      ctx.beginPath()
      ctx.arc(markerX, markerY, 4, 0, Math.PI * 2)
      ctx.fillStyle = color
      ctx.fill()
      ctx.strokeStyle = '#fff'
      ctx.lineWidth = 1.5
      ctx.stroke()

      // Label
      ctx.fillStyle = '#E5E7EB'
      ctx.font = "10px 'JetBrains Mono', monospace"
      ctx.textAlign = markerFrac > 0.7 ? 'right' : 'left'
      const labelX = markerFrac > 0.7 ? markerX - 6 : markerX + 6
      ctx.fillText(`step ${currentStep}`, labelX, markerY - 8)
    }
  }, [hyperparams, currentStep])

  return (
    <div className="metrics-chart-container">
      <h3>Learning Rate Schedule</h3>
      <canvas
        ref={canvasRef}
        style={{
          width: '100%',
          height: `${height}px`,
          background: '#111318',
          border: '1px solid rgba(255,255,255,0.06)',
          borderRadius: '10px',
        }}
      />
    </div>
  )
}
