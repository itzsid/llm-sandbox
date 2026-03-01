import type { ModelConfig, ConfigError } from '../model/schema'

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
    <div style={styles.form}>
      <div style={styles.row}>
        <Field label="Model Name" error={fieldError('name')}>
          <input
            type="text"
            style={styles.input}
            value={config.name}
            onChange={(e) => updateTop('name', e.target.value)}
          />
        </Field>
      </div>

      <div style={styles.row}>
        <Field label="Block Size (context window)" hint="8 - 1024" error={fieldError('blockSize')}>
          <div style={styles.sliderRow}>
            <input
              type="range"
              min={8}
              max={512}
              step={8}
              value={config.blockSize}
              onChange={(e) => updateTop('blockSize', parseInt(e.target.value))}
              style={styles.slider}
            />
            <input
              type="number"
              style={styles.numberInput}
              value={config.blockSize}
              min={8}
              max={1024}
              step={8}
              onChange={(e) => updateTop('blockSize', parseInt(e.target.value) || 8)}
            />
          </div>
        </Field>
      </div>

      <div style={styles.row}>
        <Field label="Number of Layers" hint="1 - 12" error={fieldError('layers')}>
          <div style={styles.sliderRow}>
            <input
              type="range"
              min={1}
              max={12}
              step={1}
              value={config.layers.length}
              onChange={(e) => handleLayerCount(parseInt(e.target.value))}
              style={styles.slider}
            />
            <span style={styles.sliderValue}>{config.layers.length}</span>
          </div>
        </Field>
      </div>

      <div style={styles.divider} />
      <div style={styles.sectionLabel}>Layer Configuration (applied to all layers)</div>

      <div style={styles.row}>
        <Field label="dModel (model dimension)" hint="16 - 512" error={fieldError('layers[0].dModel')}>
          <div style={styles.sliderRow}>
            <input
              type="range"
              min={16}
              max={512}
              step={16}
              value={layer.dModel}
              onChange={(e) => handleDModelChange(parseInt(e.target.value))}
              style={styles.slider}
            />
            <input
              type="number"
              style={styles.numberInput}
              value={layer.dModel}
              min={16}
              max={512}
              step={16}
              onChange={(e) => handleDModelChange(parseInt(e.target.value) || 16)}
            />
          </div>
        </Field>
      </div>

      <div style={styles.row}>
        <Field label="nHeads (attention heads)" hint="1 - 16, must divide dModel" error={fieldError('layers[0].nHeads')}>
          <div style={styles.sliderRow}>
            <input
              type="range"
              min={1}
              max={16}
              step={1}
              value={layer.nHeads}
              onChange={(e) => updateAllLayers('nHeads', parseInt(e.target.value))}
              style={styles.slider}
            />
            <span style={styles.sliderValue}>{layer.nHeads}</span>
          </div>
        </Field>
      </div>

      <div style={styles.row}>
        <Field label="dFF (feedforward dimension)" hint="16 - 2048" error={fieldError('layers[0].dFF')}>
          <div style={styles.sliderRow}>
            <input
              type="range"
              min={16}
              max={2048}
              step={16}
              value={layer.dFF}
              onChange={(e) => updateAllLayers('dFF', parseInt(e.target.value))}
              style={styles.slider}
            />
            <input
              type="number"
              style={styles.numberInput}
              value={layer.dFF}
              min={16}
              max={2048}
              step={16}
              onChange={(e) => updateAllLayers('dFF', parseInt(e.target.value) || 16)}
            />
          </div>
        </Field>
      </div>

      <div style={styles.row}>
        <Field label="Activation">
          <select
            style={styles.select}
            value={layer.activation}
            onChange={(e) => updateAllLayers('activation', e.target.value)}
          >
            <option value="gelu">GELU</option>
            <option value="relu">ReLU</option>
          </select>
        </Field>
      </div>

      <div style={styles.row}>
        <Field label="Dropout" hint="0 - 1">
          <div style={styles.sliderRow}>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={layer.dropout}
              onChange={(e) => updateAllLayers('dropout', parseFloat(e.target.value))}
              style={styles.slider}
            />
            <span style={styles.sliderValue}>{layer.dropout.toFixed(2)}</span>
          </div>
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
    <div style={styles.field}>
      <label style={styles.label}>
        {label}
        {hint && <span style={styles.hint}> ({hint})</span>}
      </label>
      {children}
      {error && <div style={styles.error}>{error}</div>}
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  form: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.75rem',
  },
  row: {},
  field: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.3rem',
  },
  label: {
    fontSize: '0.8rem',
    color: '#ccc',
    fontWeight: 500,
  },
  hint: {
    color: '#888',
    fontWeight: 400,
    fontSize: '0.75rem',
  },
  input: {
    width: '100%',
    padding: '0.5rem 0.75rem',
    background: '#222',
    border: '1px solid #444',
    borderRadius: '4px',
    color: '#e0e0e0',
    fontSize: '0.85rem',
    outline: 'none',
    boxSizing: 'border-box',
    fontFamily: 'inherit',
  },
  numberInput: {
    width: '80px',
    padding: '0.4rem 0.5rem',
    background: '#222',
    border: '1px solid #444',
    borderRadius: '4px',
    color: '#e0e0e0',
    fontSize: '0.85rem',
    fontFamily: "'SF Mono', 'Fira Code', monospace",
    outline: 'none',
    textAlign: 'right',
    flexShrink: 0,
  },
  select: {
    width: '100%',
    padding: '0.5rem 0.75rem',
    background: '#222',
    border: '1px solid #444',
    borderRadius: '4px',
    color: '#e0e0e0',
    fontSize: '0.85rem',
    outline: 'none',
    fontFamily: 'inherit',
  },
  sliderRow: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.75rem',
  },
  slider: {
    flex: 1,
    accentColor: '#4caf50',
  },
  sliderValue: {
    minWidth: '40px',
    textAlign: 'right',
    color: '#e0e0e0',
    fontSize: '0.85rem',
    fontFamily: "'SF Mono', 'Fira Code', monospace",
  },
  divider: {
    borderTop: '1px solid #333',
    margin: '0.5rem 0',
  },
  sectionLabel: {
    fontSize: '0.75rem',
    color: '#888',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
    marginBottom: '0.25rem',
  },
  error: {
    color: '#ff8a80',
    fontSize: '0.75rem',
  },
}
