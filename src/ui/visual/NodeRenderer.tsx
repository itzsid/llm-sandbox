import { useState } from 'react'
import type React from 'react'
import type { VisualNode } from './types'

interface NodeRendererProps {
  node: VisualNode
  selected: boolean
  onMouseDown: (e: React.MouseEvent, nodeId: string) => void
  onRemove?: (nodeId: string) => void // only for transformer blocks
  onDoubleClick?: (nodeId: string) => void
  editing?: boolean
  editValues?: Record<string, string | number>
  onEditChange?: (field: string, value: string) => void
  onEditCommit?: () => void
  onEditCancel?: () => void
}

// Parse hex color to rgba with alpha
function hexToRgba(hex: string, alpha: number): string {
  const r = parseInt(hex.slice(1, 3), 16)
  const g = parseInt(hex.slice(3, 5), 16)
  const b = parseInt(hex.slice(5, 7), 16)
  return `rgba(${r}, ${g}, ${b}, ${alpha})`
}

export const NodeRenderer: React.FC<NodeRendererProps> = ({
  node,
  selected,
  onMouseDown,
  onRemove,
  onDoubleClick,
  editing,
  editValues,
  onEditChange,
  onEditCommit,
  onEditCancel,
}) => {
  const [hovered, setHovered] = useState(false)

  // Subtle color tint for node body
  const tintColor = hexToRgba(node.color, 0.08)
  const bgColor = selected ? '#1A1D25' : '#111318'
  const strokeColor = selected ? node.color : 'rgba(255,255,255,0.08)'
  const showRemove = onRemove && (hovered || selected)

  // Format params as "key: value" lines
  const paramEntries = node.params ? Object.entries(node.params) : []
  const paramText = paramEntries.map(([k, v]) => `${k}: ${v}`).join('  ')

  // Port radius
  const portR = 5

  // Top port position: top center
  const topPortX = node.width / 2
  const topPortY = 0

  // Bottom port position: bottom center
  const bottomPortX = node.width / 2
  const bottomPortY = node.height

  return (
    <g
      transform={`translate(${node.x}, ${node.y})`}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      onDoubleClick={() => onDoubleClick?.(node.id)}
      style={{ cursor: 'grab' }}
    >
      {/* Main body — rounded rectangle */}
      <rect
        width={node.width}
        height={node.height}
        rx={8}
        ry={8}
        fill={bgColor}
        stroke={strokeColor}
        strokeWidth={selected ? 2 : 1}
        onMouseDown={(e) => onMouseDown(e, node.id)}
      />

      {/* Subtle color tint overlay */}
      <rect
        width={node.width}
        height={node.height}
        rx={8}
        ry={8}
        fill={tintColor}
        pointerEvents="none"
      />

      {/* Color accent — left border strip */}
      <rect
        x={0}
        y={4}
        width={4}
        height={node.height - 8}
        rx={2}
        ry={2}
        fill={node.color}
        pointerEvents="none"
      />

      {/* Label text — centered */}
      <text
        x={node.width / 2}
        y={paramEntries.length > 0 ? node.height / 2 - 8 : node.height / 2}
        textAnchor="middle"
        dominantBaseline="central"
        fill="#F0F0F0"
        fontSize={14}
        fontFamily="'Instrument Sans', sans-serif"
        fontWeight={600}
        pointerEvents="none"
      >
        {node.label}
      </text>

      {/* Params text — smaller, below label */}
      {paramEntries.length > 0 && (
        <text
          x={node.width / 2}
          y={node.height / 2 + 12}
          textAnchor="middle"
          dominantBaseline="central"
          fill="#9CA3AF"
          fontSize={12}
          fontFamily="'JetBrains Mono', monospace"
          pointerEvents="none"
        >
          {paramText}
        </text>
      )}

      {/* Top port (input) */}
      {node.type !== 'input' && (
        <circle
          cx={topPortX}
          cy={topPortY}
          r={portR}
          fill="#1A1D25"
          stroke={node.color}
          strokeWidth={1.5}
          opacity={0.7}
        />
      )}

      {/* Bottom port (output) */}
      {node.type !== 'output' && (
        <circle
          cx={bottomPortX}
          cy={bottomPortY}
          r={portR}
          fill="#1A1D25"
          stroke={node.color}
          strokeWidth={1.5}
          opacity={0.7}
        />
      )}

      {/* Remove button — top-right corner, only for transformer blocks */}
      {showRemove && (
        <g
          transform={`translate(${node.width - 14}, 6)`}
          style={{ cursor: 'pointer' }}
          onClick={(e) => {
            e.stopPropagation()
            onRemove!(node.id)
          }}
        >
          <circle r={9} fill="#1A1D25" stroke="rgba(255,255,255,0.08)" strokeWidth={1} />
          <text
            textAnchor="middle"
            dominantBaseline="central"
            fill="#EF4444"
            fontSize={12}
            fontFamily="'Instrument Sans', sans-serif"
            fontWeight="bold"
          >
            x
          </text>
        </g>
      )}

      {/* Inline editing via foreignObject */}
      {editing && editValues && onEditChange && onEditCommit && node.type === 'transformer_block' && (
        <foreignObject x={10} y={node.height / 2 - 10} width={node.width - 20} height={30}>
          <div style={{ display: 'flex', gap: '4px', alignItems: 'center' }}>
            {Object.entries(editValues).map(([key, val]) => (
              <input
                key={key}
                type="number"
                value={val}
                onChange={(e) => onEditChange(key, e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') onEditCommit()
                  if (e.key === 'Escape') onEditCancel?.()
                }}
                style={{
                  width: '60px',
                  padding: '2px 4px',
                  background: '#1A1D25',
                  border: '1px solid #F59E0B',
                  borderRadius: '4px',
                  color: '#F0F0F0',
                  fontSize: '11px',
                  fontFamily: "'JetBrains Mono', monospace",
                  outline: 'none',
                }}
                placeholder={key}
              />
            ))}
          </div>
        </foreignObject>
      )}
    </g>
  )
}
