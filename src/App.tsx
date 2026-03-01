import { useState, useCallback, useMemo, useEffect, useRef } from 'react'
import { GpuStatus } from './ui/GpuStatus'
import { TrainingPanel } from './ui/TrainingPanel'
import { DatasetPanel } from './ui/DatasetPanel'
import { CodeEditor } from './ui/CodeEditor'
import { NodeEditor } from './ui/visual/NodeEditor'
import { FormEditor } from './ui/FormEditor'
import { OnboardingBanner } from './ui/OnboardingBanner'
import { ConfigSummary } from './ui/ConfigSummary'
import { PlaygroundPanel } from './ui/PlaygroundPanel'
import { Trainer } from './training/trainer'
import { encodeConfigToHash, decodeConfigFromHash } from './utils/config-url'
import { loadState, saveState } from './utils/persistence'
import { loadBuiltinDataset } from './data/datasets'
import type { Dataset } from './data/datasets'
import {
  type ModelConfig,
  type ConfigError,
  validateConfig,
  estimateParamCount,
  configToText,
  textToConfig,
  PRESETS,
} from './model/schema'
import './App.css'

type ConfigSubTab = 'code' | 'visual' | 'form'

function App() {
  const [configSubTab, setConfigSubTab] = useState<ConfigSubTab>('code')
  const [modelConfig, setModelConfig] = useState<ModelConfig>(PRESETS.nano)
  const [configText, setConfigText] = useState<string>(() => configToText(PRESETS.nano))
  const [configErrors, setConfigErrors] = useState<ConfigError[]>([])
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null)
  const [trainingActive, setTrainingActive] = useState(false)
  const [restoredMsg, setRestoredMsg] = useState<string | null>(null)
  const trainingSectionRef = useRef<HTMLDivElement>(null)
  const saveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const trainerRef = useRef<Trainer | null>(null)

  const paramCount = useMemo(() => estimateParamCount(modelConfig), [modelConfig])

  // Restore state from localStorage or URL hash on mount
  useEffect(() => {
    // Check URL hash for shared config
    const hashConfig = decodeConfigFromHash(window.location.hash)
    if (hashConfig) {
      setModelConfig(hashConfig)
      setConfigText(configToText(hashConfig))
      setConfigErrors(validateConfig(hashConfig))
      // Clear the hash
      history.replaceState(null, '', window.location.pathname)
      return
    }

    // Try localStorage
    const saved = loadState()
    if (saved) {
      setConfigText(saved.configText)
      if (saved.configSubTab) setConfigSubTab(saved.configSubTab as ConfigSubTab)
      try {
        const parsed = textToConfig(saved.configText)
        const errors = validateConfig(parsed)
        setConfigErrors(errors)
        if (errors.length === 0) setModelConfig(parsed)
      } catch {
        // ignore parse errors from saved state
      }
      const ago = Math.floor((Date.now() - saved.timestamp) / 60000)
      if (ago < 60) {
        setRestoredMsg(`Restored from ${ago < 1 ? 'just now' : `${ago}m ago`}`)
        setTimeout(() => setRestoredMsg(null), 4000)
      }
    }
  }, [])

  // Auto-save to localStorage (debounced)
  useEffect(() => {
    if (saveTimerRef.current) clearTimeout(saveTimerRef.current)
    saveTimerRef.current = setTimeout(() => {
      saveState({
        configText,
        selectedDatasetId: selectedDataset?.id ?? null,
        configSubTab,
        timestamp: Date.now(),
      })
    }, 1000)
    return () => {
      if (saveTimerRef.current) clearTimeout(saveTimerRef.current)
    }
  }, [configText, selectedDataset, configSubTab])

  // Code editor changes → update modelConfig
  const handleCodeChange = useCallback((text: string) => {
    setConfigText(text)
    try {
      const parsed = textToConfig(text)
      const errors = validateConfig(parsed)
      setConfigErrors(errors)
      if (errors.length === 0) {
        setModelConfig(parsed)
      }
    } catch {
      setConfigErrors([{ path: 'JSON', message: 'Invalid JSON syntax' }])
    }
  }, [])

  // Visual editor changes → update configText
  const handleVisualChange = useCallback((config: ModelConfig) => {
    setModelConfig(config)
    setConfigText(configToText(config))
    setConfigErrors(validateConfig(config))
  }, [])

  const handleTrainingStateChange = useCallback((active: boolean) => {
    setTrainingActive(active)
  }, [])

  // Share config via URL hash
  const handleShare = useCallback(() => {
    const hash = encodeConfigToHash(modelConfig)
    const url = window.location.origin + window.location.pathname + hash
    navigator.clipboard.writeText(url).catch(() => {
      // Fallback: just update URL
    })
    history.replaceState(null, '', hash)
  }, [modelConfig])

  // Quick start: set nano preset, load Shakespeare, scroll to training
  const handleQuickStart = useCallback(async () => {
    const nano = PRESETS.nano
    setModelConfig(nano)
    setConfigText(configToText(nano))
    setConfigErrors([])
    try {
      const dataset = await loadBuiltinDataset('tiny-shakespeare')
      setSelectedDataset(dataset)
    } catch {
      // dataset loading failed, user can select manually
    }
    setTimeout(() => {
      trainingSectionRef.current?.scrollIntoView({ behavior: 'smooth' })
    }, 100)
  }, [])

  const handleStartTraining = useCallback(() => {
    trainingSectionRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [])

  return (
    <div className="app">
      <header className="header">
        <h1>LLM Sandbox</h1>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          {restoredMsg && (
            <span style={{ color: '#4caf50', fontSize: '0.75rem' }}>{restoredMsg}</span>
          )}
          <GpuStatus />
        </div>
      </header>

      <OnboardingBanner onQuickStart={handleQuickStart} />

      <main className="main">
        {/* === Configuration Section === */}
        <section className="config-section">
          <DatasetPanel selected={selectedDataset} onSelect={setSelectedDataset} />

          {/* Code / Visual / Form sub-tabs */}
          <div className="sub-tab-bar">
            <button
              className={`sub-tab-btn ${configSubTab === 'code' ? 'sub-tab-active' : ''}`}
              onClick={() => setConfigSubTab('code')}
              disabled={trainingActive}
            >
              Code
            </button>
            <button
              className={`sub-tab-btn ${configSubTab === 'visual' ? 'sub-tab-active' : ''}`}
              onClick={() => setConfigSubTab('visual')}
              disabled={trainingActive}
            >
              Visual
            </button>
            <button
              className={`sub-tab-btn ${configSubTab === 'form' ? 'sub-tab-active' : ''}`}
              onClick={() => setConfigSubTab('form')}
              disabled={trainingActive}
            >
              Form
            </button>
          </div>

          {configSubTab === 'code' && (
            <CodeEditor
              value={configText}
              onChange={handleCodeChange}
              errors={configErrors}
              paramCount={paramCount}
              onShare={handleShare}
            />
          )}

          {configSubTab === 'visual' && (
            <NodeEditor
              config={modelConfig}
              onChange={handleVisualChange}
            />
          )}

          {configSubTab === 'form' && (
            <div style={{ padding: '1rem', background: '#1a1a1a', border: '1px solid #333', borderRadius: '4px' }}>
              <FormEditor
                config={modelConfig}
                onChange={handleVisualChange}
                errors={configErrors}
              />
            </div>
          )}
        </section>

        {/* === Config Summary Bar === */}
        <ConfigSummary
          config={modelConfig}
          paramCount={paramCount}
          errors={configErrors}
          dataset={selectedDataset}
          trainingActive={trainingActive}
          onStartTraining={handleStartTraining}
        />

        {/* === Training Section === */}
        <section className="training-section" ref={trainingSectionRef}>
          <TrainingPanel
            config={modelConfig}
            dataset={selectedDataset}
            onTrainingStateChange={handleTrainingStateChange}
            trainerRef={trainerRef}
          />
        </section>

        {/* === Playground Section === */}
        <PlaygroundPanel
          trainer={trainerRef.current}
          isTraining={trainingActive}
        />
      </main>
    </div>
  )
}

export default App
