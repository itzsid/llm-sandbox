import { useRef, useEffect } from 'react'

export interface MetricsChartProps {
  data: number[]
  label: string
  color: string
  height?: number
  formatValue?: (v: number) => string
  secondaryData?: number[]
  secondaryColor?: string
  secondaryLabel?: string
}

export function MetricsChart({
  data,
  label,
  color,
  height = 120,
  formatValue = (v) => v.toFixed(2),
  secondaryData,
  secondaryColor = '#2196f3',
  secondaryLabel,
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

    // Compute Y range with padding, including secondary data if present
    const allValues = secondaryData && secondaryData.length > 0 ? [...data, ...secondaryData] : data
    const minVal = Math.min(...allValues)
    const maxVal = Math.max(...allValues)
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

    // Draw secondary data series if present
    if (secondaryData && secondaryData.length >= 2) {
      const sr = parseInt(secondaryColor.slice(1, 3), 16)
      const sg = parseInt(secondaryColor.slice(3, 5), 16)
      const sb = parseInt(secondaryColor.slice(5, 7), 16)

      // Secondary line scaled to same axis but with its own X spacing
      const toX2 = (i: number) => pad.left + (i / (secondaryData.length - 1)) * plotW

      // Gradient fill
      const gradient2 = ctx.createLinearGradient(0, pad.top, 0, pad.top + plotH)
      gradient2.addColorStop(0, `rgba(${sr}, ${sg}, ${sb}, 0.15)`)
      gradient2.addColorStop(1, `rgba(${sr}, ${sg}, ${sb}, 0.01)`)

      ctx.beginPath()
      ctx.moveTo(toX2(0), toY(secondaryData[0]))
      for (let i = 1; i < secondaryData.length; i++) {
        ctx.lineTo(toX2(i), toY(secondaryData[i]))
      }
      ctx.lineTo(toX2(secondaryData.length - 1), pad.top + plotH)
      ctx.lineTo(toX2(0), pad.top + plotH)
      ctx.closePath()
      ctx.fillStyle = gradient2
      ctx.fill()

      // Line
      ctx.beginPath()
      ctx.moveTo(toX2(0), toY(secondaryData[0]))
      for (let i = 1; i < secondaryData.length; i++) {
        ctx.lineTo(toX2(i), toY(secondaryData[i]))
      }
      ctx.strokeStyle = secondaryColor
      ctx.lineWidth = 1.5
      ctx.setLineDash([4, 3])
      ctx.stroke()
      ctx.setLineDash([])

      // Secondary dot
      const lastIdx2 = secondaryData.length - 1
      ctx.beginPath()
      ctx.arc(toX2(lastIdx2), toY(secondaryData[lastIdx2]), 3, 0, Math.PI * 2)
      ctx.fillStyle = secondaryColor
      ctx.fill()
    }

    // Legend (top-right)
    if (secondaryLabel) {
      const legendX = W - pad.right - 10
      let legendY = pad.top + 12
      ctx.textAlign = 'right'
      ctx.font = '10px -apple-system, sans-serif'

      // Primary
      ctx.fillStyle = color
      ctx.fillRect(legendX - 40, legendY - 4, 12, 2)
      ctx.fillStyle = '#aaa'
      ctx.fillText(label, legendX, legendY)
      legendY += 14

      // Secondary
      ctx.fillStyle = secondaryColor
      ctx.fillRect(legendX - 40, legendY - 4, 12, 2)
      ctx.fillStyle = '#aaa'
      ctx.fillText(secondaryLabel, legendX, legendY)
    }
  }, [data, color, formatValue, secondaryData, secondaryColor, secondaryLabel, label])

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
