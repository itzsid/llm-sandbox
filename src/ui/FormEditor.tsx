import type { ModelConfig, ConfigError } from '../model/schema'
import type { TokenizerType } from '../training/tokenizer'

interface FormEditorProps {
  config: ModelConfig
  onChange: (config: ModelConfig) => void
  errors: ConfigError[]
}

export function FormEditor({ config, onChange, errors }: FormEditorProps) {
  const layer = config.layers[0]
  if (!layer) return null

  const errorMap = new Map<string, string>()
  for (const err of errors) {
    errorMap.set(err.path, err.message)
  }

  const updateTop = (field: string, value: number | string) => {
    onChange({ ...config, [field]: value })
  }

  const updateAllLayers = (field: string, value: number | string) => {
    const newLayers = config.layers.map((l) => ({ ...l, [field]: value }))
    onChange({ ...config, layers: newLayers })
  }

  // When dModel changes, auto-adjust nHeads if needed
  const handleDModelChange = (val: number) => {
    const newLayers = config.layers.map((l) => {
      const newLayer = { ...l, dModel: val }
      if (val > 0 && l.nHeads > 0 && val % l.nHeads !== 0) {
        // Find closest valid nHeads
        for (let h = l.nHeads; h >= 1; h--) {
          if (val % h === 0) {
            newLayer.nHeads = h
            break
          }
        }
      }
      return newLayer
    })
    onChange({ ...config, layers: newLayers })
  }

  const handleLayerCount = (count: number) => {
    if (count < 1 || count > 12) return
    const current = config.layers.length
    if (count > current) {
      const extra = Array(count - current).fill(null).map(() => ({ ...config.layers[current - 1] }))
      onChange({ ...config, layers: [...config.layers, ...extra] })
    } else {
      onChange({ ...config, layers: config.layers.slice(0, count) })
    }
  }

  const fieldError = (path: string) => errorMap.get(path)

  return (
    <div className="hp-form">
      <div className="hp-grid">
        <Field label="Model Name" error={fieldError('name')}>
          <input
            type="text"
            className="hp-input hp-input-wide"
            value={config.name}
            onChange={(e) => updateTop('name', e.target.value)}
          />
        </Field>
        <Field label="Tokenizer">
          <select
            className="hp-input"
            value={config.tokenizerType ?? 'bpe-gpt2'}
            onChange={(e) => updateTop('tokenizerType', e.target.value as TokenizerType)}
          >
            <option value="bpe-gpt2">GPT-2 BPE</option>
            <option value="char">Character-level</option>
          </select>
        </Field>
        <Field label="Block Size" hint="context window, 8 - 1024" error={fieldError('blockSize')}>
          <input
            type="number"
            className="hp-input"
            value={config.blockSize}
            min={8}
            max={1024}
            step={8}
            onChange={(e) => updateTop('blockSize', parseInt(e.target.value) || 8)}
          />
        </Field>
        <Field label="Layers" hint="1 - 12" error={fieldError('layers')}>
          <input
            type="number"
            className="hp-input"
            value={config.layers.length}
            min={1}
            max={12}
            step={1}
            onChange={(e) => handleLayerCount(parseInt(e.target.value) || 1)}
          />
        </Field>
      </div>

      <div className="hp-divider" />
      <div className="hp-section-label">Layer Configuration (applied to all layers)</div>

      <div className="hp-grid">
        <Field label="dModel" hint="model dimension, 16 - 512" error={fieldError('layers[0].dModel')}>
          <input
            type="number"
            className="hp-input"
            value={layer.dModel}
            min={16}
            max={512}
            step={16}
            onChange={(e) => handleDModelChange(parseInt(e.target.value) || 16)}
          />
        </Field>
        <Field label="nHeads" hint="attention heads, must divide dModel" error={fieldError('layers[0].nHeads')}>
          <input
            type="number"
            className="hp-input"
            value={layer.nHeads}
            min={1}
            max={16}
            step={1}
            onChange={(e) => updateAllLayers('nHeads', parseInt(e.target.value) || 1)}
          />
        </Field>
        <Field label="dFF" hint="feedforward dim, 16 - 2048" error={fieldError('layers[0].dFF')}>
          <input
            type="number"
            className="hp-input"
            value={layer.dFF}
            min={16}
            max={2048}
            step={16}
            onChange={(e) => updateAllLayers('dFF', parseInt(e.target.value) || 16)}
          />
        </Field>
        <Field label="Activation">
          <select
            className="hp-input"
            value={layer.activation}
            onChange={(e) => updateAllLayers('activation', e.target.value)}
          >
            <option value="gelu">GELU</option>
            <option value="relu">ReLU</option>
          </select>
        </Field>
        <Field label="Dropout" hint="0 - 1">
          <input
            type="number"
            className="hp-input"
            value={layer.dropout}
            min={0}
            max={1}
            step={0.05}
            onChange={(e) => updateAllLayers('dropout', parseFloat(e.target.value) || 0)}
          />
        </Field>
      </div>
    </div>
  )
}

function Field({
  label,
  hint,
  error,
  children,
}: {
  label: string
  hint?: string
  error?: string
  children: React.ReactNode
}) {
  return (
    <div className="hp-field">
      <label className="hp-label">
        {label}
        {hint && <span className="hp-hint"> ({hint})</span>}
      </label>
      {children}
      {error && <div className="hp-error">{error}</div>}
    </div>
  )
}
