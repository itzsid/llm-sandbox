export interface Port {
  id: string
  side: 'top' | 'bottom' // top = input, bottom = output
}

export interface VisualNode {
  id: string
  type:
    | 'input'
    | 'token_embedding'
    | 'pos_embedding'
    | 'add'
    | 'transformer_block'
    | 'layernorm'
    | 'lm_head'
    | 'output'
  label: string
  x: number
  y: number
  width: number
  height: number
  color: string // accent color for the node
  ports: Port[]
  params?: Record<string, string | number> // displayed params (e.g., dModel: 128)
  layerIndex?: number // for transformer blocks, which layer they represent
}

export interface VisualEdge {
  id: string
  from: string // port id: "nodeId:bottom"
  to: string // port id: "nodeId:top"
  label?: string // dimension annotation e.g. "[B, T, 128]"
}

export interface VisualGraph {
  nodes: VisualNode[]
  edges: VisualEdge[]
}
