import type React from 'react'
import type { VisualNode } from './types'
import type { ModelConfig } from '../../model/schema'
import { estimateParamCount } from '../../model/schema'

interface NodeInspectorProps {
  node: VisualNode | null
  config: ModelConfig
  onChange: (config: ModelConfig) => void
}

const inspectorStyles: Record<string, React.CSSProperties> = {
  container: {
    width: 220,
    position: 'absolute' as const,
    right: 0,
    top: 0,
    bottom: 0,
    background: '#1a1a1a',
    borderLeft: '1px solid #333',
    padding: 12,
    overflowY: 'auto',
    fontFamily: "'SF Mono', 'Fira Code', 'Cascadia Code', monospace",
    color: '#e0e0e0',
    fontSize: 12,
    boxSizing: 'border-box',
  },
  placeholder: {
    color: '#666',
    fontSize: 12,
    marginTop: 20,
    textAlign: 'center' as const,
  },
  heading: {
    fontSize: 14,
    fontWeight: 'bold' as const,
    marginBottom: 4,
    color: '#e0e0e0',
  },
  typeLabel: {
    fontSize: 11,
    color: '#999',
    marginBottom: 12,
  },
  fieldGroup: {
    marginBottom: 10,
  },
  label: {
    display: 'block' as const,
    fontSize: 10,
    color: '#999',
    marginBottom: 3,
    textTransform: 'uppercase' as const,
    letterSpacing: '0.5px',
  },
  input: {
    width: '100%',
    background: '#2a2a2a',
    border: '1px solid #444',
    borderRadius: 3,
    color: '#e0e0e0',
    padding: '4px 6px',
    fontSize: 12,
    fontFamily: "'SF Mono', 'Fira Code', 'Cascadia Code', monospace",
    boxSizing: 'border-box' as const,
    outline: 'none',
  },
  inputReadonly: {
    width: '100%',
    background: '#222',
    border: '1px solid #333',
    borderRadius: 3,
    color: '#888',
    padding: '4px 6px',
    fontSize: 12,
    fontFamily: "'SF Mono', 'Fira Code', 'Cascadia Code', monospace",
    boxSizing: 'border-box' as const,
    outline: 'none',
  },
  select: {
    width: '100%',
    background: '#2a2a2a',
    border: '1px solid #444',
    borderRadius: 3,
    color: '#e0e0e0',
    padding: '4px 6px',
    fontSize: 12,
    fontFamily: "'SF Mono', 'Fira Code', 'Cascadia Code', monospace",
    boxSizing: 'border-box' as const,
    outline: 'none',
  },
  divider: {
    borderTop: '1px solid #333',
    margin: '12px 0',
  },
}

/**
 * Side panel that shows editable properties for the selected node.
 * For transformer blocks: dModel, nHeads, dFF, activation.
 * For embeddings / lm_head: shows relevant params (some read-only).
 */
export const NodeInspector: React.FC<NodeInspectorProps> = ({ node, config, onChange }) => {
  if (!node) {
    return (
      <div style={inspectorStyles.container}>
        <div style={inspectorStyles.placeholder}>Select a node to inspect</div>
      </div>
    )
  }

  // Helper: update a field on a specific transformer block layer
  const updateLayerField = (layerIndex: number, field: string, value: number | string) => {
    const newLayers = config.layers.map((layer, i) => {
      if (i !== layerIndex) return layer
      return { ...layer, [field]: value }
    })
    onChange({ ...config, layers: newLayers })
  }

  // Helper: update a top-level config field
  const updateConfigField = (field: string, value: number | string) => {
    onChange({ ...config, [field]: value })
  }

  // Resolve display values
  const vocabSize = config.vocabSize === 'auto' ? 'auto' : config.vocabSize
  const dModel = config.layers[0]?.dModel ?? 128

  const renderTransformerBlockFields = () => {
    const layerIndex = node.layerIndex ?? 0
    const layer = config.layers[layerIndex]
    if (!layer) return null

    return (
      <>
        <div style={inspectorStyles.fieldGroup}>
          <label style={inspectorStyles.label}>dModel</label>
          <input
            type="number"
            style={inspectorStyles.input}
            value={layer.dModel}
            min={1}
            step={16}
            onChange={(e) => updateLayerField(layerIndex, 'dModel', parseInt(e.target.value) || 0)}
          />
        </div>
        <div style={inspectorStyles.fieldGroup}>
          <label style={inspectorStyles.label}>nHeads</label>
          <input
            type="number"
            style={inspectorStyles.input}
            value={layer.nHeads}
            min={1}
            step={1}
            onChange={(e) => updateLayerField(layerIndex, 'nHeads', parseInt(e.target.value) || 0)}
          />
        </div>
        <div style={inspectorStyles.fieldGroup}>
          <label style={inspectorStyles.label}>dFF</label>
          <input
            type="number"
            style={inspectorStyles.input}
            value={layer.dFF}
            min={1}
            step={16}
            onChange={(e) => updateLayerField(layerIndex, 'dFF', parseInt(e.target.value) || 0)}
          />
        </div>
        <div style={inspectorStyles.fieldGroup}>
          <label style={inspectorStyles.label}>Activation</label>
          <select
            style={inspectorStyles.select}
            value={layer.activation}
            onChange={(e) =>
              updateLayerField(layerIndex, 'activation', e.target.value as 'gelu' | 'relu')
            }
          >
            <option value="gelu">GELU</option>
            <option value="relu">ReLU</option>
          </select>
        </div>
        <div style={inspectorStyles.fieldGroup}>
          <label style={inspectorStyles.label}>Dropout</label>
          <input
            type="number"
            style={inspectorStyles.input}
            value={layer.dropout}
            min={0}
            max={1}
            step={0.01}
            onChange={(e) =>
              updateLayerField(layerIndex, 'dropout', parseFloat(e.target.value) || 0)
            }
          />
        </div>
      </>
    )
  }

  const renderTokenEmbeddingFields = () => (
    <>
      <div style={inspectorStyles.fieldGroup}>
        <label style={inspectorStyles.label}>vocabSize</label>
        {vocabSize === 'auto' ? (
          <input type="text" style={inspectorStyles.inputReadonly} value="auto" readOnly />
        ) : (
          <input
            type="number"
            style={inspectorStyles.input}
            value={vocabSize}
            min={1}
            onChange={(e) =>
              updateConfigField('vocabSize', parseInt(e.target.value) || 0)
            }
          />
        )}
      </div>
      <div style={inspectorStyles.fieldGroup}>
        <label style={inspectorStyles.label}>dModel</label>
        <input type="number" style={inspectorStyles.inputReadonly} value={dModel} readOnly />
      </div>
    </>
  )

  const renderPosEmbeddingFields = () => (
    <>
      <div style={inspectorStyles.fieldGroup}>
        <label style={inspectorStyles.label}>blockSize</label>
        <input
          type="number"
          style={inspectorStyles.input}
          value={config.blockSize}
          min={1}
          step={16}
          onChange={(e) => updateConfigField('blockSize', parseInt(e.target.value) || 0)}
        />
      </div>
      <div style={inspectorStyles.fieldGroup}>
        <label style={inspectorStyles.label}>dModel</label>
        <input type="number" style={inspectorStyles.inputReadonly} value={dModel} readOnly />
      </div>
    </>
  )

  const renderLmHeadFields = () => (
    <>
      <div style={inspectorStyles.fieldGroup}>
        <label style={inspectorStyles.label}>dModel</label>
        <input type="number" style={inspectorStyles.inputReadonly} value={dModel} readOnly />
      </div>
      <div style={inspectorStyles.fieldGroup}>
        <label style={inspectorStyles.label}>vocabSize</label>
        <input
          type="text"
          style={inspectorStyles.inputReadonly}
          value={String(vocabSize)}
          readOnly
        />
      </div>
    </>
  )

  // Render the right fields based on node type
  const renderFields = () => {
    switch (node.type) {
      case 'transformer_block':
        return renderTransformerBlockFields()
      case 'token_embedding':
        return renderTokenEmbeddingFields()
      case 'pos_embedding':
        return renderPosEmbeddingFields()
      case 'lm_head':
        return renderLmHeadFields()
      default:
        return <div style={{ color: '#666', fontSize: 11 }}>No editable parameters</div>
    }
  }

  // Compute per-layer params for transformer blocks
  const totalParams = estimateParamCount(config)
  const layerParams = node.type === 'transformer_block' && node.layerIndex !== undefined
    ? (() => {
        const layer = config.layers[node.layerIndex]
        if (!layer) return 0
        const d = layer.dModel
        const ff = layer.dFF
        // Self-attention + LayerNorms + FFN
        return 4 * d * d + 4 * d + 2 * d + d * ff + ff + ff * d + d + 2 * d
      })()
    : 0
  const layerPct = totalParams > 0 ? ((layerParams / totalParams) * 100).toFixed(1) : '0'

  return (
    <div style={inspectorStyles.container}>
      <div style={inspectorStyles.heading}>{node.label}</div>
      <div style={inspectorStyles.typeLabel}>{node.type.replace(/_/g, ' ')}</div>
      <div style={inspectorStyles.divider} />
      {renderFields()}

      {/* Parameter Budget */}
      {node.type === 'transformer_block' && layerParams > 0 && (
        <>
          <div style={inspectorStyles.divider} />
          <div style={{ fontSize: 10, color: '#888', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 6 }}>
            Parameter Budget
          </div>
          <div style={{ fontSize: 12, color: '#e0e0e0' }}>
            {layerParams.toLocaleString()} params
          </div>
          <div style={{ fontSize: 11, color: '#999', marginTop: 2 }}>
            {layerPct}% of total ({totalParams.toLocaleString()})
          </div>
        </>
      )}
    </div>
  )
}
