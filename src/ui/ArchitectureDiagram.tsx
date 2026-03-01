import { useMemo } from 'react'
import type { ModelConfig } from '../model/schema'
import { estimateParamCount } from '../model/schema'

interface ArchitectureDiagramProps {
  config: ModelConfig
}

// Compact layout constants for the sidebar preview
const C_WIDTH = 160
const C_HEIGHT = 28
const C_BLOCK_HEIGHT = 36
const C_SPACING = 6
const C_CENTER_X = 130

// Node colors by type
const COLORS: Record<string, string> = {
  input: '#4B5563',
  token_embedding: '#A78BFA',
  pos_embedding: '#A78BFA',
  add: '#64748B',
  transformer_block: '#F59E0B',
  layernorm: '#FB923C',
  lm_head: '#60A5FA',
  output: '#4B5563',
}

interface CompactNode {
  label: string
  type: string
  y: number
  height: number
}

function buildCompactNodes(config: ModelConfig): CompactNode[] {
  const nodes: CompactNode[] = []
  let y = 10

  const push = (label: string, type: string, height: number) => {
    nodes.push({ label, type, y, height })
    y += height + C_SPACING
  }

  push('Input', 'input', C_HEIGHT)
  push('Token Embed', 'token_embedding', C_HEIGHT)
  push('Pos Embed', 'pos_embedding', C_HEIGHT)
  push('Add', 'add', C_HEIGHT)

  config.layers.forEach((layer, i) => {
    push(`Block ${i}  (${layer.dModel}d, ${layer.nHeads}h)`, 'transformer_block', C_BLOCK_HEIGHT)
  })

  push('LayerNorm', 'layernorm', C_HEIGHT)
  push('LM Head', 'lm_head', C_HEIGHT)
  push('Output', 'output', C_HEIGHT)

  return nodes
}

export function ArchitectureDiagram({ config }: ArchitectureDiagramProps) {
  const paramCount = useMemo(() => estimateParamCount(config), [config])
  const nodes = useMemo(() => buildCompactNodes(config), [config])

  const totalHeight = nodes.length > 0
    ? nodes[nodes.length - 1].y + nodes[nodes.length - 1].height + 10
    : 200

  const formatParams = (count: number): string => {
    if (count >= 1_000_000) return `${(count / 1_000_000).toFixed(1)}M`
    if (count >= 1_000) return `${Math.round(count / 1_000)}K`
    return count.toString()
  }

  const dModel = config.layers[0]?.dModel ?? 0
  const nHeads = config.layers[0]?.nHeads ?? 0
  const x = C_CENTER_X - C_WIDTH / 2

  return (
    <div
      style={{
        background: '#111318',
        border: '1px solid rgba(255,255,255,0.06)',
        borderRadius: 10,
        padding: 8,
      }}
    >
      <h3
        style={{
          margin: '0 0 8px 0',
          fontSize: '0.85rem',
          color: '#F0F0F0',
          fontFamily: "'Bricolage Grotesque', sans-serif",
          fontWeight: 700,
        }}
      >
        Architecture
      </h3>
      <div style={{ maxHeight: 420, overflowY: 'auto' }}>
        <svg
          viewBox={`0 0 ${C_CENTER_X * 2} ${totalHeight}`}
          preserveAspectRatio="xMidYMid meet"
          style={{ width: '100%', height: totalHeight, maxHeight: 400, display: 'block' }}
        >
          {/* Edges — simple lines between consecutive nodes */}
          {nodes.slice(0, -1).map((node, i) => {
            const next = nodes[i + 1]
            const cx = C_CENTER_X
            return (
              <line
                key={`edge-${i}`}
                x1={cx}
                y1={node.y + node.height}
                x2={cx}
                y2={next.y}
                stroke="rgba(255,255,255,0.12)"
                strokeWidth={1.5}
              />
            )
          })}

          {/* Nodes — colored rectangles with labels */}
          {nodes.map((node, i) => {
            const color = COLORS[node.type] || '#4B5563'
            return (
              <g key={i} transform={`translate(${x}, ${node.y})`}>
                <rect
                  width={C_WIDTH}
                  height={node.height}
                  rx={4}
                  ry={4}
                  fill="#111318"
                  stroke="rgba(255,255,255,0.08)"
                  strokeWidth={1}
                />
                {/* Color accent — left strip */}
                <rect
                  x={0}
                  y={3}
                  width={2.5}
                  height={node.height - 6}
                  rx={1.25}
                  fill={color}
                />
                {/* Label */}
                <text
                  x={C_WIDTH / 2}
                  y={node.height / 2}
                  textAnchor="middle"
                  dominantBaseline="central"
                  fill="#F0F0F0"
                  fontSize={11}
                  fontFamily="'Instrument Sans', sans-serif"
                >
                  {node.label}
                </text>
              </g>
            )
          })}
        </svg>
      </div>
      <div
        style={{
          fontSize: '0.75rem',
          color: '#9CA3AF',
          textAlign: 'center',
          marginTop: 6,
          fontFamily: "'JetBrains Mono', monospace",
        }}
      >
        {config.layers.length} layers, {dModel}d, {nHeads} heads | ~{formatParams(paramCount)} params
      </div>
    </div>
  )
}
