import { useState, useCallback, useMemo, useRef, useEffect } from 'react'
import type React from 'react'
import type { ModelConfig } from '../../model/schema'
import type { VisualGraph } from './types'
import { configToGraph } from './layout'
import { NodeRenderer } from './NodeRenderer'
import { EdgeRenderer } from './EdgeRenderer'
import { NodeInspector } from './NodeInspector'

interface NodeEditorProps {
  config: ModelConfig
  onChange: (config: ModelConfig) => void
}

// Zoom constraints
const MIN_ZOOM = 0.5
const MAX_ZOOM = 2.0
const ZOOM_STEP = 0.1

// SVG canvas dimensions
const CANVAS_WIDTH = 600
const CANVAS_HEIGHT = 500

const editorStyles: Record<string, React.CSSProperties> = {
  wrapper: {
    position: 'relative',
    background: '#1a1a1a',
    border: '1px solid #333',
    borderRadius: 4,
    overflow: 'hidden',
    fontFamily: "'SF Mono', 'Fira Code', 'Cascadia Code', monospace",
  },
  toolbar: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '8px 12px',
    background: '#1a1a1a',
    borderBottom: '1px solid #333',
  },
  title: {
    color: '#e0e0e0',
    fontSize: 13,
    fontWeight: 'bold',
  },
  toolbarGroup: {
    display: 'flex',
    alignItems: 'center',
    gap: 6,
  },
  button: {
    background: '#2a2a2a',
    border: '1px solid #444',
    borderRadius: 3,
    color: '#e0e0e0',
    padding: '4px 10px',
    fontSize: 11,
    fontFamily: "'SF Mono', 'Fira Code', 'Cascadia Code', monospace",
    cursor: 'pointer',
  },
  buttonPrimary: {
    background: '#2e7d32',
    border: '1px solid #4caf50',
    borderRadius: 3,
    color: '#e0e0e0',
    padding: '4px 10px',
    fontSize: 11,
    fontFamily: "'SF Mono', 'Fira Code', 'Cascadia Code', monospace",
    cursor: 'pointer',
  },
  zoomLabel: {
    color: '#999',
    fontSize: 10,
    minWidth: 36,
    textAlign: 'center' as const,
  },
  canvasContainer: {
    position: 'relative' as const,
    display: 'flex',
  },
  svgWrapper: {
    flex: 1,
    overflow: 'hidden',
    cursor: 'default',
  },
}

/**
 * Main visual editor component.
 *
 * Renders an SVG canvas showing the transformer architecture as a node graph.
 * Supports:
 *   - Pan (drag on background)
 *   - Zoom (mouse wheel)
 *   - Node drag (mousedown on node)
 *   - Node selection (click)
 *   - Add/remove transformer block layers
 *   - Node inspector for editing parameters
 */
export const NodeEditor: React.FC<NodeEditorProps> = ({ config, onChange }) => {
  // Graph state — derived from config, but with mutable node positions
  const [graph, setGraph] = useState<VisualGraph>(() => configToGraph(config))
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null)
  const [zoom, setZoom] = useState(1)
  const [panOffset, setPanOffset] = useState({ x: 0, y: 0 })

  // Dragging state
  const [dragging, setDragging] = useState<{
    type: 'node' | 'pan'
    nodeId?: string
    startX: number
    startY: number
    origX: number
    origY: number
  } | null>(null)

  const svgRef = useRef<SVGSVGElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  // Recompute graph when config changes externally
  useEffect(() => {
    setGraph(configToGraph(config))
  }, [config])

  // Compute the total SVG height based on node positions
  const svgHeight = useMemo(() => {
    if (graph.nodes.length === 0) return CANVAS_HEIGHT
    const maxY = Math.max(...graph.nodes.map((n) => n.y + n.height))
    return Math.max(CANVAS_HEIGHT, maxY + 60)
  }, [graph])

  // Selected node object
  const selectedNode = useMemo(
    () => graph.nodes.find((n) => n.id === selectedNodeId) ?? null,
    [graph, selectedNodeId]
  )

  // --- Pan & Zoom ---
  const handleWheel = useCallback(
    (e: React.WheelEvent) => {
      e.preventDefault()
      const delta = e.deltaY > 0 ? -ZOOM_STEP : ZOOM_STEP
      setZoom((z) => Math.round(Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, z + delta)) * 10) / 10)
    },
    []
  )

  const resetZoom = useCallback(() => {
    setZoom(1)
    setPanOffset({ x: 0, y: 0 })
  }, [])

  const zoomIn = useCallback(() => {
    setZoom((z) => Math.round(Math.min(MAX_ZOOM, z + ZOOM_STEP) * 10) / 10)
  }, [])

  const zoomOut = useCallback(() => {
    setZoom((z) => Math.round(Math.max(MIN_ZOOM, z - ZOOM_STEP) * 10) / 10)
  }, [])

  // --- Mouse handlers ---
  const handleBackgroundMouseDown = useCallback(
    (e: React.MouseEvent) => {
      // Only trigger pan on left click on the background
      if (e.button !== 0) return
      // Deselect any node
      setSelectedNodeId(null)
      setDragging({
        type: 'pan',
        startX: e.clientX,
        startY: e.clientY,
        origX: panOffset.x,
        origY: panOffset.y,
      })
    },
    [panOffset]
  )

  const handleNodeMouseDown = useCallback(
    (e: React.MouseEvent, nodeId: string) => {
      e.stopPropagation()
      if (e.button !== 0) return
      setSelectedNodeId(nodeId)
      const node = graph.nodes.find((n) => n.id === nodeId)
      if (!node) return
      setDragging({
        type: 'node',
        nodeId,
        startX: e.clientX,
        startY: e.clientY,
        origX: node.x,
        origY: node.y,
      })
    },
    [graph]
  )

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (!dragging) return

      const dx = (e.clientX - dragging.startX) / zoom
      const dy = (e.clientY - dragging.startY) / zoom

      if (dragging.type === 'pan') {
        setPanOffset({
          x: dragging.origX + dx,
          y: dragging.origY + dy,
        })
      } else if (dragging.type === 'node' && dragging.nodeId) {
        setGraph((prev) => ({
          ...prev,
          nodes: prev.nodes.map((n) =>
            n.id === dragging.nodeId
              ? { ...n, x: dragging.origX + dx, y: dragging.origY + dy }
              : n
          ),
        }))
      }
    },
    [dragging, zoom]
  )

  const handleMouseUp = useCallback(() => {
    setDragging(null)
  }, [])

  // --- Add / Remove layers ---
  const handleAddLayer = useCallback(() => {
    // Copy params from last layer, or use defaults
    const lastLayer = config.layers[config.layers.length - 1]
    const newLayer = lastLayer
      ? { ...lastLayer }
      : {
          type: 'transformer_block' as const,
          dModel: 128,
          nHeads: 4,
          dFF: 512,
          activation: 'gelu' as const,
          normType: 'pre' as const,
          dropout: 0.1,
        }

    const newConfig: ModelConfig = {
      ...config,
      layers: [...config.layers, newLayer],
    }
    onChange(newConfig)
  }, [config, onChange])

  const handleRemoveNode = useCallback(
    (nodeId: string) => {
      const node = graph.nodes.find((n) => n.id === nodeId)
      if (!node || node.type !== 'transformer_block' || node.layerIndex === undefined) return

      // Don't allow removing the last layer
      if (config.layers.length <= 1) return

      const newLayers = config.layers.filter((_, i) => i !== node.layerIndex)
      const newConfig: ModelConfig = { ...config, layers: newLayers }
      if (selectedNodeId === nodeId) {
        setSelectedNodeId(null)
      }
      onChange(newConfig)
    },
    [graph, config, onChange, selectedNodeId]
  )

  // --- Inspector onChange ---
  const handleInspectorChange = useCallback(
    (newConfig: ModelConfig) => {
      onChange(newConfig)
    },
    [onChange]
  )

  // SVG viewBox includes pan offset
  const viewBox = `0 0 ${CANVAS_WIDTH} ${svgHeight}`

  return (
    <div style={editorStyles.wrapper}>
      {/* Toolbar */}
      <div style={editorStyles.toolbar}>
        <span style={editorStyles.title}>Visual Editor</span>
        <div style={editorStyles.toolbarGroup}>
          <button style={editorStyles.button} onClick={zoomOut} title="Zoom out">
            -
          </button>
          <span style={editorStyles.zoomLabel}>{Math.round(zoom * 100)}%</span>
          <button style={editorStyles.button} onClick={zoomIn} title="Zoom in">
            +
          </button>
          <button style={editorStyles.button} onClick={resetZoom} title="Reset view">
            Reset
          </button>
          <button style={editorStyles.buttonPrimary} onClick={handleAddLayer}>
            + Add Layer
          </button>
        </div>
      </div>

      {/* Canvas + Inspector */}
      <div style={{ ...editorStyles.canvasContainer, height: CANVAS_HEIGHT }}>
        <div
          ref={containerRef}
          style={editorStyles.svgWrapper}
          onWheel={handleWheel}
        >
          <svg
            ref={svgRef}
            width="100%"
            height={CANVAS_HEIGHT}
            viewBox={viewBox}
            style={{ display: 'block', background: '#111' }}
            onMouseDown={handleBackgroundMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
          >
            <g transform={`translate(${panOffset.x}, ${panOffset.y}) scale(${zoom})`}>
              {/* Render edges first (behind nodes) */}
              {graph.edges.map((edge) => (
                <EdgeRenderer key={edge.id} edge={edge} nodes={graph.nodes} />
              ))}

              {/* Render nodes */}
              {graph.nodes.map((node) => (
                <NodeRenderer
                  key={node.id}
                  node={node}
                  selected={node.id === selectedNodeId}
                  onMouseDown={handleNodeMouseDown}
                  onRemove={
                    node.type === 'transformer_block' ? handleRemoveNode : undefined
                  }
                />
              ))}
            </g>
          </svg>
        </div>

        {/* Inspector panel */}
        <NodeInspector
          node={selectedNode}
          config={config}
          onChange={handleInspectorChange}
        />
      </div>
    </div>
  )
}
