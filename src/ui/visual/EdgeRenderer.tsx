import type React from 'react'
import type { VisualEdge, VisualNode } from './types'

interface EdgeRendererProps {
  edge: VisualEdge
  nodes: VisualNode[]
}

/**
 * Renders an edge as a cubic bezier curve between two node ports.
 *
 * edge.from = "nodeId:bottom" => source is bottom-center of source node
 * edge.to   = "nodeId:top"    => target is top-center of target node
 */
export const EdgeRenderer: React.FC<EdgeRendererProps> = ({ edge, nodes }) => {
  // Parse port IDs to extract node IDs
  const fromNodeId = edge.from.split(':')[0]
  const toNodeId = edge.to.split(':')[0]

  const sourceNode = nodes.find((n) => n.id === fromNodeId)
  const targetNode = nodes.find((n) => n.id === toNodeId)

  if (!sourceNode || !targetNode) return null

  // Source point: bottom center of source node
  const sx = sourceNode.x + sourceNode.width / 2
  const sy = sourceNode.y + sourceNode.height

  // Target point: top center of target node
  const tx = targetNode.x + targetNode.width / 2
  const ty = targetNode.y

  // Control point offset — proportional to vertical distance, clamped
  const dy = Math.abs(ty - sy)
  const cpOffset = Math.max(30, Math.min(dy * 0.4, 80))

  // Cubic bezier path
  const path = `M ${sx} ${sy} C ${sx} ${sy + cpOffset}, ${tx} ${ty - cpOffset}, ${tx} ${ty}`

  // Arrow dot at target
  const dotR = 3

  // Midpoint of bezier for label placement
  const midX = (sx + tx) / 2
  const midY = (sy + ty) / 2

  return (
    <g>
      <path
        d={path}
        fill="none"
        stroke="rgba(255,255,255,0.12)"
        strokeWidth={2}
        strokeLinecap="round"
      />
      {/* Small arrowhead dot at the target end */}
      <circle
        cx={tx}
        cy={ty}
        r={dotR}
        fill="rgba(255,255,255,0.12)"
      />
      {/* Dimension label at midpoint */}
      {edge.label && (
        <text
          x={midX + 8}
          y={midY}
          fill="#4B5563"
          fontSize={9}
          fontFamily="'Instrument Sans', sans-serif"
          dominantBaseline="central"
        >
          {edge.label}
        </text>
      )}
    </g>
  )
}
