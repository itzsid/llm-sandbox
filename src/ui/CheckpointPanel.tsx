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
import {
  saveCloudCheckpoint,
  listCloudCheckpoints,
  loadCloudCheckpoint,
  deleteCloudCheckpoint,
} from '../firebase/cloudStorage'
import type { CloudCheckpointMeta } from '../firebase/cloudStorage'
import type { TransformerParams } from '../model/transformer'
import type { TransformerConfig } from '../model/config'
import type { User } from 'firebase/auth'
import type { TokenizerState } from '../training/tokenizer'
import type { TrainingHyperparams } from '../training/trainer'

export interface CheckpointPanelProps {
  params: TransformerParams | null
  config: TransformerConfig
  step: number
  lossHistory: number[]
  valLossHistory: number[]
  hyperparams: TrainingHyperparams
  tokenizerState: TokenizerState
  datasetId: string
  onLoad: (checkpoint: Checkpoint) => void
  disabled: boolean
  user: User | null
}

interface CheckpointEntry {
  name: string
  step: number
  savedAt: number
  datasetId: string
}

type StorageTab = 'local' | 'cloud'

export function CheckpointPanel({
  params,
  config,
  step,
  lossHistory,
  valLossHistory,
  hyperparams,
  tokenizerState,
  datasetId,
  onLoad,
  disabled,
  user,
}: CheckpointPanelProps) {
  const [activeTab, setActiveTab] = useState<StorageTab>('local')
  const [checkpoints, setCheckpoints] = useState<CheckpointEntry[]>([])
  const [cloudCheckpoints, setCloudCheckpoints] = useState<CloudCheckpointMeta[]>([])
  const [saveName, setSaveName] = useState('')
  const [showSaveInput, setShowSaveInput] = useState(false)
  const [saving, setSaving] = useState(false)
  const [loading, setLoading] = useState<string | null>(null)
  const [cloudSaving, setCloudSaving] = useState(false)
  const [cloudLoading, setCloudLoading] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [cloudListLoading, setCloudListLoading] = useState(false)
  const [successMsg, setSuccessMsg] = useState<string | null>(null)

  const refreshList = useCallback(async () => {
    try {
      const list = await listCheckpoints()
      setCheckpoints(list)
    } catch (e) {
      console.error('Failed to list checkpoints:', e)
    }
  }, [])

  const refreshCloudList = useCallback(async () => {
    if (!user) return
    setCloudListLoading(true)
    try {
      const list = await listCloudCheckpoints(user.uid)
      setCloudCheckpoints(list)
    } catch (e) {
      console.error('Failed to list cloud checkpoints:', e)
    } finally {
      setCloudListLoading(false)
    }
  }, [user])

  useEffect(() => {
    refreshList()
  }, [refreshList])

  useEffect(() => {
    if (user && activeTab === 'cloud') {
      refreshCloudList()
    }
  }, [user, activeTab, refreshCloudList])

  // Reset to local tab when user signs out
  useEffect(() => {
    if (!user && activeTab === 'cloud') {
      setActiveTab('local')
    }
  }, [user, activeTab])

  const handleSave = useCallback(async () => {
    if (!params || !saveName.trim()) return
    const trimmed = saveName.trim()
    if (checkpoints.some((cp) => cp.name === trimmed)) {
      if (!confirm(`A checkpoint named "${trimmed}" already exists. Overwrite?`)) return
    }
    setSaving(true)
    setError(null)
    try {
      const serialized = await serializeParams(params)
      const checkpoint: Checkpoint = {
        version: 3,
        name: saveName.trim(),
        config,
        step,
        lossHistory,
        params: serialized,
        tokenizer: tokenizerState,
        datasetId,
        savedAt: Date.now(),
        hyperparams,
        valLossHistory: valLossHistory.length > 0 ? valLossHistory : undefined,
      }
      await saveCheckpoint(checkpoint)
      setSaveName('')
      setShowSaveInput(false)
      await refreshList()
      setError(null)
      setSuccessMsg('Checkpoint saved!')
      setTimeout(() => setSuccessMsg(null), 3000)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setSaving(false)
    }
  }, [params, saveName, config, step, lossHistory, valLossHistory, hyperparams, tokenizerState, datasetId, refreshList, checkpoints])

  const handleCloudSave = useCallback(async () => {
    if (!params || !saveName.trim() || !user) return
    const trimmed = saveName.trim()
    if (cloudCheckpoints.some((cp) => cp.name === trimmed)) {
      if (!confirm(`A cloud checkpoint named "${trimmed}" already exists. Overwrite?`)) return
    }
    setCloudSaving(true)
    setError(null)
    try {
      const serialized = await serializeParams(params)
      const checkpoint: Checkpoint = {
        version: 3,
        name: saveName.trim(),
        config,
        step,
        lossHistory,
        params: serialized,
        tokenizer: tokenizerState,
        datasetId,
        savedAt: Date.now(),
        hyperparams,
        valLossHistory: valLossHistory.length > 0 ? valLossHistory : undefined,
      }
      await saveCloudCheckpoint(user.uid, checkpoint)
      setSaveName('')
      setShowSaveInput(false)
      await refreshCloudList()
      setError(null)
      setSuccessMsg('Checkpoint saved to cloud!')
      setTimeout(() => setSuccessMsg(null), 3000)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setCloudSaving(false)
    }
  }, [params, saveName, config, step, lossHistory, valLossHistory, hyperparams, tokenizerState, datasetId, user, refreshCloudList, cloudCheckpoints])

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

  const handleCloudLoad = useCallback(async (name: string) => {
    if (!user) return
    setCloudLoading(name)
    setError(null)
    try {
      const checkpoint = await loadCloudCheckpoint(user.uid, name)
      onLoad(checkpoint)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setCloudLoading(null)
    }
  }, [user, onLoad])

  const handleDelete = useCallback(async (name: string) => {
    if (!confirm(`Delete checkpoint "${name}"? This cannot be undone.`)) return
    setError(null)
    try {
      await deleteCheckpoint(name)
      await refreshList()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }, [refreshList])

  const handleCloudDelete = useCallback(async (name: string) => {
    if (!user) return
    if (!confirm(`Delete cloud checkpoint "${name}"? This cannot be undone.`)) return
    setError(null)
    try {
      await deleteCloudCheckpoint(user.uid, name)
      await refreshCloudList()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }, [user, refreshCloudList])

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

  const formatSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  }

  const isCloud = activeTab === 'cloud'
  const isSaving = isCloud ? cloudSaving : saving
  const handleSaveAction = isCloud ? handleCloudSave : handleSave

  return (
    <div className="checkpoint-panel" style={styles.container}>
      <h3 style={styles.heading}>Checkpoints</h3>

      {/* Local/Cloud tabs */}
      {user && (
        <div style={styles.tabBar}>
          <button
            onClick={() => { setActiveTab('local'); setError(null); setSuccessMsg(null) }}
            style={{ ...styles.tabBtn, ...(activeTab === 'local' ? styles.tabActive : {}) }}
          >
            Local
          </button>
          <button
            onClick={() => { setActiveTab('cloud'); setError(null); setSuccessMsg(null) }}
            style={{ ...styles.tabBtn, ...(activeTab === 'cloud' ? styles.tabActive : {}) }}
          >
            Cloud
          </button>
        </div>
      )}

      {error && <div style={styles.error}>{error}</div>}
      {successMsg && <div style={{ color: 'var(--green, #22c55e)', fontSize: '0.85rem', marginBottom: '0.75rem' }}>{successMsg}</div>}

      <div style={styles.actions}>
        {showSaveInput ? (
          <div style={styles.saveRow}>
            <input
              type="text"
              value={saveName}
              onChange={(e) => setSaveName(e.target.value)}
              placeholder="Checkpoint name..."
              style={styles.input}
              disabled={disabled || isSaving}
              onKeyDown={(e) => {
                if (e.key === 'Enter') handleSaveAction()
                if (e.key === 'Escape') { setShowSaveInput(false); setSaveName('') }
              }}
              autoFocus
            />
            <button
              onClick={handleSaveAction}
              disabled={disabled || isSaving || !saveName.trim() || !params}
              style={{ ...styles.btn, ...styles.btnSave }}
            >
              {isSaving ? 'Saving...' : 'Save'}
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
              {isCloud ? 'Save to Cloud' : 'Save Checkpoint'}
            </button>
            {!isCloud && (
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
            )}
          </div>
        )}
      </div>

      {/* Local checkpoint list */}
      {!isCloud && (
        <>
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
        </>
      )}

      {/* Cloud checkpoint list */}
      {isCloud && (
        <>
          {cloudListLoading ? (
            <div style={{ color: 'var(--text-3)', padding: '1rem', textAlign: 'center' }}>Loading checkpoints...</div>
          ) : cloudCheckpoints.length === 0 ? (
            <div style={styles.emptyMsg}>No cloud checkpoints</div>
          ) : (
            <div style={styles.list}>
              {cloudCheckpoints.map((cp) => (
                <div key={cp.name} style={styles.item}>
                  <div style={styles.itemInfo}>
                    <span style={styles.itemName}>{cp.name}</span>
                    <span style={styles.itemMeta}>
                      Step {cp.step} | {cp.configSummary} | {formatSize(cp.size)} | {formatDate(cp.savedAt)}
                    </span>
                  </div>
                  <div style={styles.itemActions}>
                    <button
                      onClick={() => handleCloudLoad(cp.name)}
                      disabled={disabled || cloudLoading === cp.name}
                      style={{ ...styles.btnSmall, ...styles.btnLoad }}
                    >
                      {cloudLoading === cp.name ? '...' : 'Load'}
                    </button>
                    <button
                      onClick={() => handleCloudDelete(cp.name)}
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
        </>
      )}
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    marginTop: '1.5rem',
    padding: '1rem',
    background: 'var(--bg-surface)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-md)',
  },
  heading: {
    fontSize: '0.85rem',
    color: 'var(--text-3)',
    margin: '0 0 0.75rem',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
    fontFamily: 'var(--font-body)',
  },
  tabBar: {
    display: 'flex',
    gap: '0.25rem',
    marginBottom: '0.75rem',
    background: '#111',
    borderRadius: '4px',
    padding: '0.2rem',
  },
  tabBtn: {
    flex: 1,
    padding: '0.3rem 0.5rem',
    border: 'none',
    borderRadius: '3px',
    background: 'transparent',
    color: '#888',
    fontSize: '0.75rem',
    cursor: 'pointer',
    fontFamily: 'inherit',
  },
  tabActive: {
    background: '#333',
    color: '#e0e0e0',
  },
  error: {
    background: 'rgba(239, 68, 68, 0.1)',
    border: '1px solid var(--red)',
    color: '#FCA5A5',
    padding: '0.5rem',
    borderRadius: 'var(--radius-sm)',
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
    background: 'var(--bg-elevated)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-sm)',
    color: 'var(--text-1)',
    fontSize: '0.85rem',
    fontFamily: 'inherit',
    outline: 'none',
  },
  btn: {
    padding: '0.4rem 0.8rem',
    border: 'none',
    borderRadius: 'var(--radius-sm)',
    fontSize: '0.8rem',
    cursor: 'pointer',
    fontFamily: 'var(--font-body)',
    fontWeight: 500,
    transition: 'all 0.2s ease',
  },
  btnSave: {
    background: 'var(--amber)',
    color: '#000',
  },
  btnCancel: {
    background: 'transparent',
    border: '1px solid var(--border)',
    color: 'var(--text-2)',
  },
  btnImport: {
    background: 'transparent',
    border: '1px solid var(--border)',
    color: 'var(--text-2)',
    display: 'inline-flex',
    alignItems: 'center',
    cursor: 'pointer',
  },
  btnDisabled: {
    opacity: 0.5,
    cursor: 'not-allowed',
  },
  emptyMsg: {
    color: 'var(--text-3)',
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
    background: 'var(--bg-elevated)',
    borderRadius: 'var(--radius-sm)',
    border: '1px solid var(--border)',
  },
  itemInfo: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.15rem',
    minWidth: 0,
    flex: 1,
  },
  itemName: {
    color: 'var(--text-1)',
    fontSize: '0.85rem',
    fontWeight: 600,
  },
  itemMeta: {
    color: 'var(--text-3)',
    fontSize: '0.7rem',
    fontFamily: 'var(--font-mono)',
  },
  itemActions: {
    display: 'flex',
    gap: '0.3rem',
    flexShrink: 0,
  },
  btnSmall: {
    padding: '0.25rem 0.5rem',
    border: 'none',
    borderRadius: 'var(--radius-sm)',
    fontSize: '0.7rem',
    cursor: 'pointer',
    fontFamily: 'var(--font-body)',
    transition: 'all 0.2s ease',
  },
  btnLoad: {
    background: 'transparent',
    border: '1px solid var(--border)',
    color: 'var(--text-2)',
  },
  btnExport: {
    background: 'transparent',
    border: '1px solid var(--border)',
    color: 'var(--text-2)',
  },
  btnDelete: {
    background: 'var(--red)',
    color: '#fff',
  },
}
