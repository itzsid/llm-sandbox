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
}) => {
  const [hovered, setHovered] = useState(false)

  // Subtle color tint for node body
  const tintColor = hexToRgba(node.color, 0.08)
  const bgColor = selected ? '#2a2a2a' : '#1e1e1e'
  const strokeColor = selected ? node.color : '#444'
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
        fill="#e0e0e0"
        fontSize={15}
        fontFamily="'SF Mono', 'Fira Code', 'Cascadia Code', monospace"
        fontWeight={500}
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
          fill="#999"
          fontSize={13}
          fontFamily="'SF Mono', 'Fira Code', 'Cascadia Code', monospace"
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
          fill="#555"
          stroke={node.color}
          strokeWidth={1.5}
        />
      )}

      {/* Bottom port (output) */}
      {node.type !== 'output' && (
        <circle
          cx={bottomPortX}
          cy={bottomPortY}
          r={portR}
          fill="#555"
          stroke={node.color}
          strokeWidth={1.5}
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
          <circle r={9} fill="#333" stroke="#666" strokeWidth={1} />
          <text
            textAnchor="middle"
            dominantBaseline="central"
            fill="#f44336"
            fontSize={12}
            fontFamily="'SF Mono', 'Fira Code', 'Cascadia Code', monospace"
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
                onKeyDown={(e) => { if (e.key === 'Enter') onEditCommit() }}
                style={{
                  width: '60px',
                  padding: '2px 4px',
                  background: '#2a2a2a',
                  border: '1px solid #4caf50',
                  borderRadius: '3px',
                  color: '#e0e0e0',
                  fontSize: '11px',
                  fontFamily: "'SF Mono', monospace",
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
