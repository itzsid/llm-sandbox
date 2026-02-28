import type { ModelConfig, LayerConfig } from '../../model/schema'
import type { VisualGraph, VisualNode, VisualEdge, Port } from './types'

// Node colors by type
const NODE_COLORS: Record<string, string> = {
  input: '#666',
  token_embedding: '#9c27b0', // purple
  pos_embedding: '#9c27b0', // purple
  add: '#607d8b', // gray-blue
  transformer_block: '#4caf50', // green
  layernorm: '#ff9800', // orange
  lm_head: '#2196f3', // blue
  output: '#666', // gray
}

const NODE_WIDTH = 280
const NODE_HEIGHT_NORMAL = 64
const NODE_HEIGHT_BLOCK = 90 // transformer blocks are taller to show params
const VERTICAL_SPACING = 60
const CENTER_X = 350 // center for ~700px wide canvas
const START_Y = 30

function makeNode(
  id: string,
  type: VisualNode['type'],
  label: string,
  y: number,
  params?: Record<string, string | number>,
  layerIndex?: number
): VisualNode {
  const height = type === 'transformer_block' ? NODE_HEIGHT_BLOCK : NODE_HEIGHT_NORMAL
  const ports: Port[] = [
    { id: `${id}:top`, side: 'top' },
    { id: `${id}:bottom`, side: 'bottom' },
  ]

  return {
    id,
    type,
    label,
    x: CENTER_X - NODE_WIDTH / 2,
    y,
    width: NODE_WIDTH,
    height,
    color: NODE_COLORS[type] || '#666',
    ports,
    params,
    layerIndex,
  }
}

function makeEdge(fromNodeId: string, toNodeId: string): VisualEdge {
  return {
    id: `${fromNodeId}:bottom->${toNodeId}:top`,
    from: `${fromNodeId}:bottom`,
    to: `${toNodeId}:top`,
  }
}

/**
 * Convert a ModelConfig into a visual graph representation.
 *
 * Creates a linear chain:
 *   Input -> Token Embedding -> Pos Embedding -> Add ->
 *   [Transformer Block x N] -> Final LayerNorm -> LM Head -> Output
 *
 * Layout: center-aligned vertically, spaced 80px apart.
 */
export function configToGraph(config: ModelConfig): VisualGraph {
  const nodes: VisualNode[] = []
  const edges: VisualEdge[] = []
  let y = START_Y

  // Derive dModel from first layer (or default to 128)
  const firstLayer = config.layers[0]
  const dModel = firstLayer ? firstLayer.dModel : 128
  const vocabSize = config.vocabSize === 'auto' ? 'auto' : config.vocabSize

  // 1. Input node
  const inputNode = makeNode('input', 'input', 'Input', y)
  nodes.push(inputNode)
  y += NODE_HEIGHT_NORMAL + VERTICAL_SPACING

  // 2. Token Embedding
  const tokenEmbNode = makeNode('token_embedding', 'token_embedding', 'Token Embedding', y, {
    vocabSize,
    dModel,
  })
  nodes.push(tokenEmbNode)
  edges.push(makeEdge('input', 'token_embedding'))
  y += NODE_HEIGHT_NORMAL + VERTICAL_SPACING

  // 3. Positional Embedding
  const posEmbNode = makeNode('pos_embedding', 'pos_embedding', 'Pos Embedding', y, {
    blockSize: config.blockSize,
    dModel,
  })
  nodes.push(posEmbNode)
  edges.push(makeEdge('token_embedding', 'pos_embedding'))
  y += NODE_HEIGHT_NORMAL + VERTICAL_SPACING

  // 4. Add (combine token + pos embeddings)
  const addNode = makeNode('add', 'add', 'Add', y)
  nodes.push(addNode)
  edges.push(makeEdge('pos_embedding', 'add'))
  y += NODE_HEIGHT_NORMAL + VERTICAL_SPACING

  // 5. Transformer Blocks
  let prevNodeId = 'add'
  config.layers.forEach((layer, index) => {
    const blockId = `transformer_block_${index}`
    const blockNode = makeNode(
      blockId,
      'transformer_block',
      `Block ${index}`,
      y,
      {
        dModel: layer.dModel,
        nHeads: layer.nHeads,
        dFF: layer.dFF,
      },
      index
    )
    nodes.push(blockNode)
    edges.push(makeEdge(prevNodeId, blockId))
    prevNodeId = blockId
    y += NODE_HEIGHT_BLOCK + VERTICAL_SPACING
  })

  // 6. Final LayerNorm
  const lnNode = makeNode('layernorm', 'layernorm', 'Layer Norm', y, { dModel })
  nodes.push(lnNode)
  edges.push(makeEdge(prevNodeId, 'layernorm'))
  y += NODE_HEIGHT_NORMAL + VERTICAL_SPACING

  // 7. LM Head (linear projection back to vocab)
  const lmHeadNode = makeNode('lm_head', 'lm_head', 'LM Head', y, {
    dModel,
    vocabSize,
  })
  nodes.push(lmHeadNode)
  edges.push(makeEdge('layernorm', 'lm_head'))
  y += NODE_HEIGHT_NORMAL + VERTICAL_SPACING

  // 8. Output node
  const outputNode = makeNode('output', 'output', 'Output', y)
  nodes.push(outputNode)
  edges.push(makeEdge('lm_head', 'output'))

  return { nodes, edges }
}

/**
 * Convert a visual graph back to a ModelConfig.
 *
 * Extracts transformer_block nodes from the graph, reads their params,
 * and builds the layers array. Preserves name, vocabSize, blockSize,
 * and tieWeights from the currentConfig.
 */
export function graphToConfig(graph: VisualGraph, currentConfig: ModelConfig): ModelConfig {
  // Collect transformer block nodes, sorted by layerIndex
  const blockNodes = graph.nodes
    .filter((n) => n.type === 'transformer_block')
    .sort((a, b) => (a.layerIndex ?? 0) - (b.layerIndex ?? 0))

  const layers: LayerConfig[] = blockNodes.map((node) => {
    const params = node.params || {}
    return {
      type: 'transformer_block' as const,
      dModel: typeof params.dModel === 'number' ? params.dModel : currentConfig.layers[0]?.dModel ?? 128,
      nHeads: typeof params.nHeads === 'number' ? params.nHeads : currentConfig.layers[0]?.nHeads ?? 4,
      dFF: typeof params.dFF === 'number' ? params.dFF : currentConfig.layers[0]?.dFF ?? 512,
      activation: (currentConfig.layers[node.layerIndex ?? 0]?.activation ?? 'gelu') as 'gelu' | 'relu',
      normType: 'pre' as const,
      dropout: currentConfig.layers[node.layerIndex ?? 0]?.dropout ?? 0.1,
    }
  })

  return {
    ...currentConfig,
    layers,
  }
}
