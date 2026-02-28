import { useRef, useEffect } from 'react'

export interface MetricsChartProps {
  data: number[]
  label: string
  color: string
  height?: number
  formatValue?: (v: number) => string
}

export function MetricsChart({
  data,
  label,
  color,
  height = 120,
  formatValue = (v) => v.toFixed(2),
}: MetricsChartProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || data.length < 2) return
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
    const minVal = Math.min(...data)
    const maxVal = Math.max(...data)
    const yPad = Math.max((maxVal - minVal) * 0.1, 0.01)
    const yMin = minVal - yPad
    const yMax = maxVal + yPad

    const pad = { top: 8, right: 8, bottom: 20, left: 40 }
    const plotW = W - pad.left - pad.right
    const plotH = H - pad.top - pad.bottom

    const toX = (i: number) => pad.left + (i / (data.length - 1)) * plotW
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
      ctx.fillText(formatValue(v), pad.left - 4, y + 3)
    }

    // X axis label
    ctx.textAlign = 'center'
    ctx.fillStyle = '#555'
    ctx.fillText(`${data.length} steps`, W / 2, H - 2)

    // Parse color for gradient (convert hex to rgba)
    const r = parseInt(color.slice(1, 3), 16)
    const g = parseInt(color.slice(3, 5), 16)
    const b = parseInt(color.slice(5, 7), 16)

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

    // Current value dot
    const lastIdx = data.length - 1
    ctx.beginPath()
    ctx.arc(toX(lastIdx), toY(data[lastIdx]), 3, 0, Math.PI * 2)
    ctx.fillStyle = color
    ctx.fill()
  }, [data, color, formatValue])

  return (
    <div className="metrics-chart-container">
      <h3>{label}</h3>
      <canvas
        ref={canvasRef}
        style={{
          width: '100%',
          height: `${height}px`,
          background: '#1a1a1a',
          border: '1px solid #333',
          borderRadius: '4px',
        }}
      />
    </div>
  )
}
