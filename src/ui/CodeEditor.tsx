import { useRef, useCallback, useMemo, useState } from 'react'
import { highlightJSON } from './code-highlight'
import {
  type ModelConfig,
  type ConfigError,
  estimateParamCount,
  configToText,
  PRESETS,
} from '../model/schema'

interface CodeEditorProps {
  value: string
  onChange: (text: string) => void
  errors: ConfigError[]
  paramCount: number
  onShare?: () => void
}

function formatParamCount(count: number): string {
  if (count >= 1_000_000) {
    return `~${(count / 1_000_000).toFixed(1)}M params`
  }
  if (count >= 1_000) {
    return `~${(count / 1_000).toFixed(0)}K params`
  }
  return `~${count} params`
}

const PRESET_ENTRIES: { key: string; label: string; config: ModelConfig }[] = [
  { key: 'tiny', label: 'Tiny', config: PRESETS.tiny },
  { key: 'nano', label: 'Nano', config: PRESETS.nano },
  { key: 'micro', label: 'Micro', config: PRESETS.micro },
]

// Pre-compute preset param counts for button labels
const PRESET_LABELS: Record<string, string> = {}
const PRESET_TOOLTIPS: Record<string, string> = {}
for (const entry of PRESET_ENTRIES) {
  PRESET_LABELS[entry.key] = `${entry.label} (${formatParamCount(estimateParamCount(entry.config))})`
  const l = entry.config.layers[0]
  PRESET_TOOLTIPS[entry.key] = `${entry.config.layers.length} layers, dModel=${l.dModel}, nHeads=${l.nHeads}, dFF=${l.dFF}, blockSize=${entry.config.blockSize}`
}

// Map error paths to approximate line numbers in the JSON text
function mapErrorsToLines(text: string, errors: ConfigError[]): Map<number, ConfigError[]> {
  const lineMap = new Map<number, ConfigError[]>()
  const lines = text.split('\n')

  for (const err of errors) {
    // Try to find the line containing the error path
    const pathParts = err.path.split('.')
    const lastPart = pathParts[pathParts.length - 1]
    // Search for the key in the JSON
    let foundLine = -1
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i]
      if (line.includes(`"${lastPart}"`)) {
        // Check if it's within the right array context
        if (err.path.includes('[')) {
          const match = err.path.match(/\[(\d+)\]/)
          if (match) {
            const idx = parseInt(match[1])
            // Count opening braces to find which block we're in
            let braceCount = 0
            let blockIdx = -1
            for (let j = 0; j <= i; j++) {
              if (lines[j].includes('{')) braceCount++
              if (lines[j].includes('}')) braceCount--
              if (lines[j].includes('{') && j > 0) blockIdx++
            }
            // Rough heuristic: skip if block index doesn't match
            if (blockIdx >= 0 && Math.floor(blockIdx / 1) !== idx + 1) continue
          }
        }
        foundLine = i
        break
      }
    }
    if (foundLine === -1) foundLine = 0
    const existing = lineMap.get(foundLine) || []
    existing.push(err)
    lineMap.set(foundLine, existing)
  }

  return lineMap
}

export function CodeEditor({ value, onChange, errors, paramCount, onShare }: CodeEditorProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const preRef = useRef<HTMLPreElement>(null)
  const gutterRef = useRef<HTMLDivElement>(null)
  const [shareMsg, setShareMsg] = useState<string | null>(null)

  // Determine which preset (if any) matches the current editor value
  const activePresetKey = useMemo(() => {
    const trimmed = value.trim()
    for (const entry of PRESET_ENTRIES) {
      if (configToText(entry.config).trim() === trimmed) return entry.key
    }
    return null
  }, [value])

  // Sync scroll between textarea, pre, and gutter
  const handleScroll = useCallback(() => {
    const textarea = textareaRef.current
    const pre = preRef.current
    const gutter = gutterRef.current
    if (textarea && pre) {
      pre.scrollTop = textarea.scrollTop
      pre.scrollLeft = textarea.scrollLeft
    }
    if (textarea && gutter) {
      gutter.scrollTop = textarea.scrollTop
    }
  }, [])

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLTextAreaElement>) => {
      onChange(e.target.value)
    },
    [onChange],
  )

  const handlePresetClick = useCallback(
    (key: string) => {
      const preset = PRESETS[key]
      if (preset) {
        onChange(configToText(preset))
      }
    },
    [onChange],
  )

  // Memoize highlighted HTML
  const highlightedHTML = useMemo(() => highlightJSON(value), [value])

  // Map errors to line numbers for gutter markers
  const errorLineMap = useMemo(() => mapErrorsToLines(value, errors), [value, errors])
  const lineCount = value.split('\n').length

  // Handle tab key in textarea (insert two spaces instead of focus change)
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === 'Tab') {
        e.preventDefault()
        const textarea = e.currentTarget
        const start = textarea.selectionStart
        const end = textarea.selectionEnd
        const newValue = value.substring(0, start) + '  ' + value.substring(end)
        onChange(newValue)
        // Restore cursor position after React re-renders
        requestAnimationFrame(() => {
          textarea.selectionStart = textarea.selectionEnd = start + 2
        })
      }
    },
    [value, onChange],
  )

  return (
    <div style={styles.container}>
      {/* Header with presets and param count */}
      <div style={styles.header}>
        <div style={styles.presetRow}>
          {PRESET_ENTRIES.map((entry) => {
            const isActive = activePresetKey === entry.key
            return (
              <button
                key={entry.key}
                style={isActive ? { ...styles.presetButton, ...styles.presetButtonActive } : styles.presetButton}
                title={PRESET_TOOLTIPS[entry.key]}
                onClick={() => handlePresetClick(entry.key)}
                onMouseEnter={(e) => {
                  if (!isActive) {
                    ;(e.target as HTMLElement).style.borderColor = '#666'
                    ;(e.target as HTMLElement).style.background = '#333'
                  }
                }}
                onMouseLeave={(e) => {
                  if (!isActive) {
                    ;(e.target as HTMLElement).style.borderColor = '#444'
                    ;(e.target as HTMLElement).style.background = '#2a2a2a'
                  }
                }}
              >
                {PRESET_LABELS[entry.key]}
              </button>
            )
          })}
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <div style={styles.paramCount}>{formatParamCount(paramCount)}</div>
          {onShare && (
            <button
              style={styles.shareButton}
              onClick={() => {
                onShare()
                setShareMsg('Copied!')
                setTimeout(() => setShareMsg(null), 2000)
              }}
            >
              {shareMsg ?? 'Share'}
            </button>
          )}
        </div>
      </div>

      {/* Editor area with gutter */}
      <div style={styles.editorWrapper}>
        {/* Error gutter */}
        <div ref={gutterRef} style={styles.gutter}>
          {Array.from({ length: lineCount }, (_, i) => {
            const lineErrors = errorLineMap.get(i)
            return (
              <div
                key={i}
                style={styles.gutterLine}
                title={lineErrors ? lineErrors.map((e) => `${e.message}${e.suggestion ? `\n${e.suggestion}` : ''}`).join('\n') : undefined}
              >
                {lineErrors ? (
                  <span style={styles.gutterMarker}>!</span>
                ) : null}
              </div>
            )
          })}
        </div>
        <div style={{ position: 'relative', flex: 1, minHeight: '300px' }}>
          <pre
            ref={preRef}
            style={styles.pre}
            dangerouslySetInnerHTML={{ __html: highlightedHTML + '\n' }}
          />
          <textarea
            ref={textareaRef}
            style={styles.textarea}
            value={value}
            onChange={handleChange}
            onScroll={handleScroll}
            onKeyDown={handleKeyDown}
            spellCheck={false}
            autoComplete="off"
            autoCorrect="off"
            autoCapitalize="off"
          />
        </div>
      </div>

      {/* Error list */}
      {errors.length > 0 && (
        <div style={styles.errorList}>
          {errors.map((err, i) => (
            <div key={i} style={styles.errorItem}>
              <span style={styles.errorPath}>{err.path}:</span> {err.message}
              {err.suggestion && (
                <div style={styles.errorSuggestion}>{err.suggestion}</div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ---- Styles ----

const monoFont = "'SF Mono', 'Fira Code', 'Cascadia Code', monospace"

const sharedEditorStyle: React.CSSProperties = {
  fontFamily: monoFont,
  fontSize: '13px',
  lineHeight: '1.5',
  padding: '12px',
  border: 'none',
  margin: 0,
  whiteSpace: 'pre',
  wordWrap: 'normal',
  overflowWrap: 'normal',
  tabSize: 2,
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0',
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '8px 12px',
    background: '#1a1a1a',
    borderTopLeftRadius: '4px',
    borderTopRightRadius: '4px',
    border: '1px solid #333',
    borderBottom: 'none',
  },
  presetRow: {
    display: 'flex',
    gap: '6px',
    flexWrap: 'wrap' as const,
  },
  presetButton: {
    background: '#2a2a2a',
    border: '1px solid #444',
    borderRadius: '12px',
    color: '#ccc',
    padding: '3px 10px',
    fontSize: '0.75rem',
    fontFamily: monoFont,
    cursor: 'pointer',
    transition: 'border-color 0.15s, background 0.15s',
  },
  presetButtonActive: {
    background: '#333',
    borderColor: '#4caf50',
    color: '#4caf50',
  },
  paramCount: {
    color: '#888',
    fontSize: '0.8rem',
    fontFamily: monoFont,
    whiteSpace: 'nowrap' as const,
  },
  editorWrapper: {
    position: 'relative' as const,
    display: 'flex',
    minHeight: '300px',
    background: '#1a1a1a',
    border: '1px solid #333',
    borderBottomLeftRadius: '4px',
    borderBottomRightRadius: '4px',
    overflow: 'hidden',
  },
  gutter: {
    width: '24px',
    background: '#181818',
    borderRight: '1px solid #333',
    overflow: 'hidden',
    flexShrink: 0,
    paddingTop: '12px',
  },
  gutterLine: {
    height: '19.5px', // matches lineHeight 1.5 * 13px
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: '10px',
  },
  gutterMarker: {
    color: '#f44336',
    fontWeight: 'bold' as const,
    fontSize: '11px',
    cursor: 'help',
  },
  pre: {
    ...sharedEditorStyle,
    position: 'absolute' as const,
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    pointerEvents: 'none' as const,
    zIndex: 0,
    overflow: 'auto',
    background: 'transparent',
    color: '#e0e0e0',
  },
  textarea: {
    ...sharedEditorStyle,
    position: 'relative' as const,
    width: '100%',
    minHeight: '300px',
    background: 'transparent',
    color: 'transparent',
    caretColor: '#e0e0e0',
    zIndex: 1,
    outline: 'none',
    resize: 'vertical' as const,
    boxSizing: 'border-box' as const,
  },
  errorList: {
    padding: '8px 12px',
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '4px',
  },
  errorItem: {
    color: '#ff8a80',
    fontSize: '0.8rem',
    fontFamily: monoFont,
  },
  errorPath: {
    color: '#ffab91',
    fontWeight: 600,
  },
  errorSuggestion: {
    color: '#64b5f6',
    fontSize: '0.75rem',
    marginTop: '2px',
    fontFamily: monoFont,
  },
  shareButton: {
    background: '#2a2a2a',
    border: '1px solid #444',
    borderRadius: '12px',
    color: '#64b5f6',
    padding: '3px 10px',
    fontSize: '0.75rem',
    cursor: 'pointer',
    transition: 'border-color 0.15s, background 0.15s',
  },
}
