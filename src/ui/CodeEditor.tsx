import { useRef, useCallback, useMemo } from 'react'
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
for (const entry of PRESET_ENTRIES) {
  PRESET_LABELS[entry.key] = `${entry.label} (${formatParamCount(estimateParamCount(entry.config))})`
}

export function CodeEditor({ value, onChange, errors, paramCount }: CodeEditorProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const preRef = useRef<HTMLPreElement>(null)

  // Sync scroll between textarea and pre
  const handleScroll = useCallback(() => {
    const textarea = textareaRef.current
    const pre = preRef.current
    if (textarea && pre) {
      pre.scrollTop = textarea.scrollTop
      pre.scrollLeft = textarea.scrollLeft
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
          {PRESET_ENTRIES.map((entry) => (
            <button
              key={entry.key}
              style={styles.presetButton}
              onClick={() => handlePresetClick(entry.key)}
              onMouseEnter={(e) => {
                ;(e.target as HTMLElement).style.borderColor = '#666'
                ;(e.target as HTMLElement).style.background = '#333'
              }}
              onMouseLeave={(e) => {
                ;(e.target as HTMLElement).style.borderColor = '#444'
                ;(e.target as HTMLElement).style.background = '#2a2a2a'
              }}
            >
              {PRESET_LABELS[entry.key]}
            </button>
          ))}
        </div>
        <div style={styles.paramCount}>{formatParamCount(paramCount)}</div>
      </div>

      {/* Editor area */}
      <div style={styles.editorWrapper}>
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

      {/* Error list */}
      {errors.length > 0 && (
        <div style={styles.errorList}>
          {errors.map((err, i) => (
            <div key={i} style={styles.errorItem}>
              <span style={styles.errorPath}>{err.path}:</span> {err.message}
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
  paramCount: {
    color: '#888',
    fontSize: '0.8rem',
    fontFamily: monoFont,
    whiteSpace: 'nowrap' as const,
  },
  editorWrapper: {
    position: 'relative' as const,
    minHeight: '300px',
    background: '#1a1a1a',
    border: '1px solid #333',
    borderBottomLeftRadius: '4px',
    borderBottomRightRadius: '4px',
    overflow: 'hidden',
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
}
