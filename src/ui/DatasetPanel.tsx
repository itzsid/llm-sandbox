import { useState, useEffect, useCallback } from 'react'
import { getBuiltinDatasets, loadBuiltinDataset, createCustomDataset, type Dataset } from '../data/datasets'

export interface DatasetPanelProps {
  selected: Dataset | null
  onSelect: (dataset: Dataset) => void
}

export function DatasetPanel({ selected, onSelect }: DatasetPanelProps) {
  const [builtinDatasets] = useState(() => getBuiltinDatasets())
  const [loadingId, setLoadingId] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleSelectBuiltin = useCallback(async (id: string) => {
    setLoadingId(id)
    setError(null)
    try {
      const dataset = await loadBuiltinDataset(id)
      onSelect(dataset)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoadingId(null)
    }
  }, [onSelect])

  const handleUpload = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setError(null)
    try {
      const text = await file.text()
      if (text.length === 0) {
        setError('File is empty')
        return
      }
      const name = file.name.replace(/\.txt$/i, '')
      const dataset = createCustomDataset(name, text)
      onSelect(dataset)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
    // Reset file input
    e.target.value = ''
  }, [onSelect])

  // Auto-select tiny-shakespeare if nothing is selected
  useEffect(() => {
    if (!selected && builtinDatasets.length > 0) {
      handleSelectBuiltin(builtinDatasets[0].id)
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="dataset-panel" style={styles.container}>
      <h3 style={styles.heading}>Dataset</h3>

      {error && <div style={styles.error}>{error}</div>}

      <div style={styles.list}>
        {builtinDatasets.map((ds) => (
          <label key={ds.id} style={styles.radioLabel}>
            <input
              type="radio"
              name="dataset"
              checked={selected?.id === ds.id}
              onChange={() => handleSelectBuiltin(ds.id)}
              disabled={loadingId !== null}
              style={styles.radio}
            />
            <div style={styles.datasetInfo}>
              <span style={styles.datasetName}>
                {ds.name}
                {loadingId === ds.id && <span style={styles.loadingDot}> loading...</span>}
              </span>
              <span style={styles.datasetDesc}>{ds.description}</span>
              <span style={styles.datasetSize}>{ds.size}</span>
            </div>
          </label>
        ))}

        {selected && selected.id.startsWith('custom-') && (
          <label style={styles.radioLabel}>
            <input
              type="radio"
              name="dataset"
              checked
              readOnly
              style={styles.radio}
            />
            <div style={styles.datasetInfo}>
              <span style={styles.datasetName}>{selected.name}</span>
              <span style={styles.datasetDesc}>{selected.description}</span>
              <span style={styles.datasetSize}>{selected.size}</span>
            </div>
          </label>
        )}
      </div>

      <div style={styles.uploadRow}>
        <label style={styles.uploadBtn}>
          Upload Custom (.txt)
          <input
            type="file"
            accept=".txt"
            onChange={handleUpload}
            style={{ display: 'none' }}
          />
        </label>
      </div>

      {selected && (
        <div style={styles.selectedInfo}>
          <span style={styles.selectedLabel}>Selected:</span> {selected.name} ({selected.size})
        </div>
      )}
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    marginTop: '1.5rem',
    padding: '1rem',
    background: '#1a1a1a',
    border: '1px solid #333',
    borderRadius: '4px',
  },
  heading: {
    fontSize: '0.85rem',
    color: '#888',
    margin: '0 0 0.75rem',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
  },
  error: {
    background: '#3e1a1a',
    border: '1px solid #f44336',
    color: '#ff8a80',
    padding: '0.5rem',
    borderRadius: '4px',
    marginBottom: '0.75rem',
    fontSize: '0.8rem',
  },
  list: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.5rem',
    marginBottom: '0.75rem',
  },
  radioLabel: {
    display: 'flex',
    alignItems: 'flex-start',
    gap: '0.5rem',
    padding: '0.5rem',
    background: '#222',
    borderRadius: '4px',
    border: '1px solid #333',
    cursor: 'pointer',
  },
  radio: {
    marginTop: '0.2rem',
    accentColor: '#4caf50',
  },
  datasetInfo: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.1rem',
  },
  datasetName: {
    color: '#e0e0e0',
    fontSize: '0.85rem',
    fontWeight: 600,
  },
  datasetDesc: {
    color: '#999',
    fontSize: '0.75rem',
    lineHeight: '1.3',
  },
  datasetSize: {
    color: '#666',
    fontSize: '0.7rem',
  },
  loadingDot: {
    color: '#ff9800',
    fontWeight: 400,
    fontSize: '0.75rem',
  },
  uploadRow: {
    marginBottom: '0.75rem',
  },
  uploadBtn: {
    display: 'inline-flex',
    alignItems: 'center',
    padding: '0.4rem 0.8rem',
    background: '#2196f3',
    color: '#fff',
    border: 'none',
    borderRadius: '4px',
    fontSize: '0.8rem',
    cursor: 'pointer',
    fontFamily: 'inherit',
  },
  selectedInfo: {
    color: '#aaa',
    fontSize: '0.75rem',
    paddingTop: '0.25rem',
    borderTop: '1px solid #333',
  },
  selectedLabel: {
    color: '#4caf50',
    fontWeight: 600,
  },
}
