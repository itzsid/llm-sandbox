import { useState, useEffect, useCallback } from 'react'
import {
  saveCheckpoint,
  loadCheckpoint,
  listCheckpoints,
  deleteCheckpoint,
  exportCheckpoint,
  importCheckpoint,
  serializeParams,
  type Checkpoint,
} from '../storage/checkpoint'
import type { TransformerParams } from '../model/transformer'
import type { TransformerConfig } from '../model/config'

export interface CheckpointPanelProps {
  params: TransformerParams | null
  config: TransformerConfig
  step: number
  lossHistory: number[]
  vocab: string[]
  datasetId: string
  onLoad: (checkpoint: Checkpoint) => void
  disabled: boolean
}

interface CheckpointEntry {
  name: string
  step: number
  savedAt: number
  datasetId: string
}

export function CheckpointPanel({
  params,
  config,
  step,
  lossHistory,
  vocab,
  datasetId,
  onLoad,
  disabled,
}: CheckpointPanelProps) {
  const [checkpoints, setCheckpoints] = useState<CheckpointEntry[]>([])
  const [saveName, setSaveName] = useState('')
  const [showSaveInput, setShowSaveInput] = useState(false)
  const [saving, setSaving] = useState(false)
  const [loading, setLoading] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const refreshList = useCallback(async () => {
    try {
      const list = await listCheckpoints()
      setCheckpoints(list)
    } catch (e) {
      console.error('Failed to list checkpoints:', e)
    }
  }, [])

  useEffect(() => {
    refreshList()
  }, [refreshList])

  const handleSave = useCallback(async () => {
    if (!params || !saveName.trim()) return
    setSaving(true)
    setError(null)
    try {
      const serialized = await serializeParams(params)
      const checkpoint: Checkpoint = {
        version: 1,
        name: saveName.trim(),
        config,
        step,
        lossHistory,
        params: serialized,
        vocab,
        datasetId,
        savedAt: Date.now(),
      }
      await saveCheckpoint(checkpoint)
      setSaveName('')
      setShowSaveInput(false)
      await refreshList()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setSaving(false)
    }
  }, [params, saveName, config, step, lossHistory, vocab, datasetId, refreshList])

  const handleLoad = useCallback(async (name: string) => {
    setLoading(name)
    setError(null)
    try {
      const checkpoint = await loadCheckpoint(name)
      onLoad(checkpoint)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(null)
    }
  }, [onLoad])

  const handleDelete = useCallback(async (name: string) => {
    setError(null)
    try {
      await deleteCheckpoint(name)
      await refreshList()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }, [refreshList])

  const handleExport = useCallback(async (name: string) => {
    setError(null)
    try {
      const checkpoint = await loadCheckpoint(name)
      const blob = exportCheckpoint(checkpoint)
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${name}.llmsb`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }, [])

  const handleImport = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setError(null)
    try {
      const checkpoint = await importCheckpoint(file)
      await saveCheckpoint(checkpoint)
      await refreshList()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
    // Reset file input
    e.target.value = ''
  }, [refreshList])

  const formatDate = (timestamp: number) => {
    const d = new Date(timestamp)
    return d.toLocaleDateString() + ' ' + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  }

  return (
    <div className="checkpoint-panel" style={styles.container}>
      <h3 style={styles.heading}>Checkpoints</h3>

      {error && <div style={styles.error}>{error}</div>}

      <div style={styles.actions}>
        {showSaveInput ? (
          <div style={styles.saveRow}>
            <input
              type="text"
              value={saveName}
              onChange={(e) => setSaveName(e.target.value)}
              placeholder="Checkpoint name..."
              style={styles.input}
              disabled={disabled || saving}
              onKeyDown={(e) => {
                if (e.key === 'Enter') handleSave()
                if (e.key === 'Escape') { setShowSaveInput(false); setSaveName('') }
              }}
              autoFocus
            />
            <button
              onClick={handleSave}
              disabled={disabled || saving || !saveName.trim() || !params}
              style={{ ...styles.btn, ...styles.btnSave }}
            >
              {saving ? 'Saving...' : 'Save'}
            </button>
            <button
              onClick={() => { setShowSaveInput(false); setSaveName('') }}
              style={{ ...styles.btn, ...styles.btnCancel }}
            >
              Cancel
            </button>
          </div>
        ) : (
          <div style={styles.buttonRow}>
            <button
              onClick={() => setShowSaveInput(true)}
              disabled={disabled || !params}
              style={{ ...styles.btn, ...styles.btnSave }}
            >
              Save Checkpoint
            </button>
            <label style={{ ...styles.btn, ...styles.btnImport, ...(disabled ? styles.btnDisabled : {}) }}>
              Import
              <input
                type="file"
                accept=".llmsb"
                onChange={handleImport}
                disabled={disabled}
                style={{ display: 'none' }}
              />
            </label>
          </div>
        )}
      </div>

      {checkpoints.length === 0 ? (
        <div style={styles.emptyMsg}>No saved checkpoints</div>
      ) : (
        <div style={styles.list}>
          {checkpoints.map((cp) => (
            <div key={cp.name} style={styles.item}>
              <div style={styles.itemInfo}>
                <span style={styles.itemName}>{cp.name}</span>
                <span style={styles.itemMeta}>
                  Step {cp.step} | {formatDate(cp.savedAt)}
                </span>
              </div>
              <div style={styles.itemActions}>
                <button
                  onClick={() => handleLoad(cp.name)}
                  disabled={disabled || loading === cp.name}
                  style={{ ...styles.btnSmall, ...styles.btnLoad }}
                >
                  {loading === cp.name ? '...' : 'Load'}
                </button>
                <button
                  onClick={() => handleExport(cp.name)}
                  disabled={disabled}
                  style={{ ...styles.btnSmall, ...styles.btnExport }}
                >
                  Export
                </button>
                <button
                  onClick={() => handleDelete(cp.name)}
                  disabled={disabled}
                  style={{ ...styles.btnSmall, ...styles.btnDelete }}
                >
                  Delete
                </button>
              </div>
            </div>
          ))}
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
  actions: {
    marginBottom: '0.75rem',
  },
  saveRow: {
    display: 'flex',
    gap: '0.5rem',
    alignItems: 'center',
  },
  buttonRow: {
    display: 'flex',
    gap: '0.5rem',
  },
  input: {
    flex: 1,
    padding: '0.4rem 0.6rem',
    background: '#2a2a2a',
    border: '1px solid #444',
    borderRadius: '4px',
    color: '#e0e0e0',
    fontSize: '0.85rem',
    fontFamily: 'inherit',
    outline: 'none',
  },
  btn: {
    padding: '0.4rem 0.8rem',
    border: 'none',
    borderRadius: '4px',
    fontSize: '0.8rem',
    cursor: 'pointer',
    fontFamily: 'inherit',
  },
  btnSave: {
    background: '#4caf50',
    color: '#fff',
  },
  btnCancel: {
    background: '#555',
    color: '#ccc',
  },
  btnImport: {
    background: '#2196f3',
    color: '#fff',
    display: 'inline-flex',
    alignItems: 'center',
    cursor: 'pointer',
  },
  btnDisabled: {
    opacity: 0.5,
    cursor: 'not-allowed',
  },
  emptyMsg: {
    color: '#666',
    fontSize: '0.8rem',
    fontStyle: 'italic',
  },
  list: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.5rem',
  },
  item: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '0.5rem 0.6rem',
    background: '#222',
    borderRadius: '4px',
    border: '1px solid #333',
  },
  itemInfo: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.15rem',
  },
  itemName: {
    color: '#e0e0e0',
    fontSize: '0.85rem',
    fontWeight: 600,
  },
  itemMeta: {
    color: '#888',
    fontSize: '0.7rem',
  },
  itemActions: {
    display: 'flex',
    gap: '0.3rem',
  },
  btnSmall: {
    padding: '0.25rem 0.5rem',
    border: 'none',
    borderRadius: '3px',
    fontSize: '0.7rem',
    cursor: 'pointer',
    fontFamily: 'inherit',
  },
  btnLoad: {
    background: '#4caf50',
    color: '#fff',
  },
  btnExport: {
    background: '#ff9800',
    color: '#fff',
  },
  btnDelete: {
    background: '#f44336',
    color: '#fff',
  },
}
