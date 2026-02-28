import { useMemo } from 'react'
import type { ModelConfig } from '../model/schema'
import { configToGraph } from './visual/layout'
import { estimateParamCount } from '../model/schema'
import { NodeRenderer } from './visual/NodeRenderer'
import { EdgeRenderer } from './visual/EdgeRenderer'

interface ArchitectureDiagramProps {
  config: ModelConfig
}

export function ArchitectureDiagram({ config }: ArchitectureDiagramProps) {
  const graph = useMemo(() => configToGraph(config), [config])
  const paramCount = useMemo(() => estimateParamCount(config), [config])

  // Calculate viewBox to fit all nodes with padding
  const bounds = useMemo(() => {
    if (graph.nodes.length === 0) {
      return { minX: 0, minY: 0, width: 400, height: 200 }
    }
    let minX = Infinity
    let minY = Infinity
    let maxX = -Infinity
    let maxY = -Infinity
    for (const n of graph.nodes) {
      minX = Math.min(minX, n.x)
      minY = Math.min(minY, n.y)
      maxX = Math.max(maxX, n.x + n.width)
      maxY = Math.max(maxY, n.y + n.height)
    }
    const pad = 20
    return {
      minX: minX - pad,
      minY: minY - pad,
      width: maxX - minX + 2 * pad,
      height: maxY - minY + 2 * pad,
    }
  }, [graph])

  const formatParams = (count: number): string => {
    if (count >= 1_000_000) return `${(count / 1_000_000).toFixed(1)}M`
    if (count >= 1_000) return `${Math.round(count / 1_000)}K`
    return count.toString()
  }

  const dModel = config.layers[0]?.dModel ?? 0
  const nHeads = config.layers[0]?.nHeads ?? 0

  const noopMouseDown = () => {}

  return (
    <div
      style={{
        background: '#1a1a1a',
        border: '1px solid #333',
        borderRadius: 4,
        padding: 8,
      }}
    >
      <h3
        style={{
          margin: '0 0 8px 0',
          fontSize: '0.85rem',
          color: '#e0e0e0',
          fontFamily: "'SF Mono', 'Fira Code', 'Cascadia Code', monospace",
        }}
      >
        Architecture
      </h3>
      <svg
        viewBox={`${bounds.minX} ${bounds.minY} ${bounds.width} ${bounds.height}`}
        preserveAspectRatio="xMidYMid meet"
        style={{ width: '100%', height: 250, display: 'block' }}
      >
        {graph.edges.map(edge => (
          <EdgeRenderer key={edge.id} edge={edge} nodes={graph.nodes} />
        ))}
        {graph.nodes.map(node => (
          <NodeRenderer
            key={node.id}
            node={node}
            selected={false}
            onMouseDown={noopMouseDown}
          />
        ))}
      </svg>
      <div
        style={{
          fontSize: '0.75rem',
          color: '#888',
          textAlign: 'center',
          marginTop: 6,
          fontFamily: "'SF Mono', 'Fira Code', 'Cascadia Code', monospace",
        }}
      >
        {config.layers.length} layers, {dModel}d, {nHeads} heads | ~{formatParams(paramCount)} params
      </div>
    </div>
  )
}
