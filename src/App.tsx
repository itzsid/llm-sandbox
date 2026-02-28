import { useState, useCallback, useMemo } from 'react'
import { GpuStatus } from './ui/GpuStatus'
import { TrainingPanel } from './ui/TrainingPanel'
import { DatasetPanel } from './ui/DatasetPanel'
import { CodeEditor } from './ui/CodeEditor'
import { NodeEditor } from './ui/visual/NodeEditor'
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

type Tab = 'configure' | 'train'
type ConfigSubTab = 'code' | 'visual'

function App() {
  const [tab, setTab] = useState<Tab>('configure')
  const [configSubTab, setConfigSubTab] = useState<ConfigSubTab>('code')
  const [modelConfig, setModelConfig] = useState<ModelConfig>(PRESETS.nano)
  const [configText, setConfigText] = useState<string>(() => configToText(PRESETS.nano))
  const [configErrors, setConfigErrors] = useState<ConfigError[]>([])
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null)
  const [, setTrainingActive] = useState(false)

  const paramCount = useMemo(() => estimateParamCount(modelConfig), [modelConfig])

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

  return (
    <div className="app">
      <header className="header">
        <h1>LLM Sandbox</h1>
        <GpuStatus />
      </header>

      {/* Tab Bar */}
      <nav className="tab-bar">
        <button
          className={`tab-btn ${tab === 'configure' ? 'tab-active' : ''}`}
          onClick={() => setTab('configure')}
        >
          Configure
        </button>
        <button
          className={`tab-btn ${tab === 'train' ? 'tab-active' : ''}`}
          onClick={() => setTab('train')}
          disabled={configErrors.length > 0}
          title={configErrors.length > 0 ? 'Fix config errors before training' : undefined}
        >
          Train
        </button>
      </nav>

      <main className="main">
        {tab === 'configure' && (
          <div className="configure-tab">
            <DatasetPanel selected={selectedDataset} onSelect={setSelectedDataset} />

            {/* Code / Visual sub-tabs */}
            <div className="sub-tab-bar">
              <button
                className={`sub-tab-btn ${configSubTab === 'code' ? 'sub-tab-active' : ''}`}
                onClick={() => setConfigSubTab('code')}
              >
                Code
              </button>
              <button
                className={`sub-tab-btn ${configSubTab === 'visual' ? 'sub-tab-active' : ''}`}
                onClick={() => setConfigSubTab('visual')}
              >
                Visual
              </button>
            </div>

            {configSubTab === 'code' && (
              <CodeEditor
                value={configText}
                onChange={handleCodeChange}
                errors={configErrors}
                paramCount={paramCount}
              />
            )}

            {configSubTab === 'visual' && (
              <NodeEditor
                config={modelConfig}
                onChange={handleVisualChange}
              />
            )}
          </div>
        )}

        {tab === 'train' && (
          <TrainingPanel
            config={modelConfig}
            dataset={selectedDataset}
            onTrainingStateChange={handleTrainingStateChange}
          />
        )}
      </main>
    </div>
  )
}

export default App
