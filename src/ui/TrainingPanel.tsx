import { useState, useRef, useCallback, useEffect } from 'react'
import { Trainer, type TrainingMetrics } from '../training/trainer'
import { MetricsChart } from './MetricsChart'
import { CheckpointPanel } from './CheckpointPanel'
import { ArchitectureDiagram } from './ArchitectureDiagram'
import { toLegacyConfig } from '../model/schema'
import type { ModelConfig } from '../model/schema'
import type { Dataset } from '../data/datasets'
import type { Checkpoint } from '../storage/checkpoint'

function formatTokenCount(count: number): string {
  if (count >= 1_000_000) return `${(count / 1_000_000).toFixed(1)}M`
  if (count >= 1_000) return `${(count / 1_000).toFixed(0)}K`
  return String(count)
}

function formatElapsed(seconds: number): string {
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = seconds % 60
  if (h > 0) return `${h}h ${m}m ${s}s`
  if (m > 0) return `${m}m ${s}s`
  return `${s}s`
}

interface TrainingPanelProps {
  config: ModelConfig
  dataset: Dataset | null
  onTrainingStateChange: (active: boolean) => void
  trainerRef?: React.MutableRefObject<Trainer | null>
}

export function TrainingPanel({ config, dataset, onTrainingStateChange, trainerRef: externalTrainerRef }: TrainingPanelProps) {
  const [status, setStatus] = useState<'idle' | 'initializing' | 'training' | 'stopped'>('idle')
  const [metrics, setMetrics] = useState<TrainingMetrics | null>(null)
  const [lossHistory, setLossHistory] = useState<number[]>([])
  const [valLossHistory, setValLossHistory] = useState<number[]>([])
  const [tokensPerSecHistory, setTokensPerSecHistory] = useState<number[]>([])
  const [sampleText, setSampleText] = useState<string>('')
  const [error, setError] = useState<string | null>(null)
  const [temperature, setTemperature] = useState(0.8)
  const [maxTokens, setMaxTokens] = useState(100)
  const [prompt, setPrompt] = useState('')
  const [generating, setGenerating] = useState(false)
  const [trainingStartTime, setTrainingStartTime] = useState<number | null>(null)
  const [elapsedTime, setElapsedTime] = useState<string>('')
  const trainerRef = useRef<Trainer | null>(null)

  // Sync trainer to external ref
  const setTrainer = useCallback((trainer: Trainer | null) => {
    trainerRef.current = trainer
    if (externalTrainerRef) externalTrainerRef.current = trainer
  }, [externalTrainerRef])

  const isTraining = status === 'training'

  useEffect(() => {
    onTrainingStateChange(isTraining)
  }, [isTraining, onTrainingStateChange])

  // Elapsed time ticker
  useEffect(() => {
    if (!isTraining || !trainingStartTime) return
    const interval = setInterval(() => {
      const elapsed = Math.floor((Date.now() - trainingStartTime) / 1000)
      setElapsedTime(formatElapsed(elapsed))
    }, 1000)
    return () => clearInterval(interval)
  }, [isTraining, trainingStartTime])

  const handleStart = useCallback(async () => {
    if (!dataset) {
      setError('Select a dataset first')
      return
    }
    try {
      setError(null)
      setStatus('initializing')

      const legacyConfig = toLegacyConfig(config)
      const trainer = new Trainer(legacyConfig)
      setTrainer(trainer)
      await trainer.init(dataset.text)

      setStatus('training')
      setTrainingStartTime(Date.now())
      await trainer.train(
        (m) => {
          setMetrics(m)
          setLossHistory((prev) => [...prev.slice(-199), m.loss])
          setTokensPerSecHistory((prev) => [...prev.slice(-199), m.tokensPerSec])
          if (m.valLoss !== undefined) {
            setValLossHistory((prev) => [...prev.slice(-49), m.valLoss!])
          }
        },
        (text) => {
          setSampleText(text)
        },
      )
      setStatus('stopped')
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setStatus('idle')
    }
  }, [config, dataset])

  const handleStop = useCallback(() => {
    trainerRef.current?.stop()
    setStatus('stopped')
  }, [])

  const handleGenerate = useCallback(async () => {
    const trainer = trainerRef.current
    if (!trainer || !trainer.params) return
    setGenerating(true)
    try {
      const text = await trainer.generateSample(maxTokens, temperature, prompt || undefined)
      setSampleText(text)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setGenerating(false)
    }
  }, [temperature, maxTokens, prompt])

  const handleCheckpointLoad = useCallback(async (checkpoint: Checkpoint) => {
    if (!dataset) {
      setError('Select a dataset first')
      return
    }
    try {
      setError(null)
      setStatus('initializing')

      const legacyConfig = checkpoint.config
      const trainer = new Trainer(legacyConfig)
      setTrainer(trainer)
      await trainer.loadFromCheckpoint(
        checkpoint.params,
        checkpoint.config,
        checkpoint.step,
        checkpoint.lossHistory,
        checkpoint.vocab,
        dataset.text,
      )

      setLossHistory(checkpoint.lossHistory.slice(-200))
      setMetrics({
        step: checkpoint.step,
        loss: checkpoint.lossHistory[checkpoint.lossHistory.length - 1] ?? 0,
        tokensPerSec: 0,
        learningRate: 3e-4,
      })
      setStatus('stopped')
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setStatus('idle')
    }
  }, [dataset])

  const trainer = trainerRef.current

  return (
    <div className="training-panel">
      {/* Full-width: metrics strip + charts */}
      <div className="metrics-strip">
        {(status === 'idle' || status === 'stopped') && (
          <button onClick={handleStart} className="btn-start">
            {status === 'stopped' ? 'Resume Training' : 'Start Training'}
          </button>
        )}
        {status === 'training' && (
          <button onClick={handleStop} className="btn-stop">
            Stop
          </button>
        )}
        {status === 'initializing' && (
          <button disabled className="btn-disabled">
            Initializing...
          </button>
        )}
        {!dataset && status === 'idle' && (
          <span className="no-dataset-hint">Select a dataset in Configure tab</span>
        )}
        {metrics && (
          <div className="metrics">
            <div className="metric">
              <span className="metric-label">Step</span>
              <span className="metric-value">{metrics.step}</span>
            </div>
            <div className="metric">
              <span className="metric-label">Loss</span>
              <span className="metric-value">{metrics.loss.toFixed(4)}</span>
            </div>
            <div className="metric">
              <span className="metric-label">Tokens/s</span>
              <span className="metric-value">{metrics.tokensPerSec.toFixed(0)}</span>
            </div>
            {metrics.valLoss !== undefined && (
              <div className="metric">
                <span className="metric-label">Val Loss</span>
                <span className="metric-value">{metrics.valLoss.toFixed(4)}</span>
              </div>
            )}
            <div className="metric">
              <span className="metric-label">LR</span>
              <span className="metric-value">{metrics.learningRate.toExponential(1)}</span>
            </div>
            <div className="metric">
              <span className="metric-label">Tokens Trained</span>
              <span className="metric-value">{formatTokenCount(metrics.step * 4 * config.blockSize)}</span>
            </div>
            {elapsedTime && (
              <div className="metric">
                <span className="metric-label">Elapsed</span>
                <span className="metric-value">{elapsedTime}</span>
              </div>
            )}
          </div>
        )}
      </div>

      {error && <div className="error-msg">{error}</div>}

      {lossHistory.length > 1 && (
        <MetricsChart
          data={lossHistory}
          label="Train Loss"
          color="#4caf50"
          height={160}
          formatValue={(v) => v.toFixed(2)}
          secondaryData={valLossHistory.length > 1 ? valLossHistory : undefined}
          secondaryColor="#2196f3"
          secondaryLabel="Val Loss"
        />
      )}

      {tokensPerSecHistory.length > 1 && (
        <MetricsChart
          data={tokensPerSecHistory}
          label="Tokens/sec"
          color="#ff9800"
          height={100}
          formatValue={(v) => v.toFixed(0)}
        />
      )}

      {/* 2-column: generation + sidebar */}
      <div className="training-layout">
        <div className="training-main">
          {/* Generation controls */}
          <div className="gen-controls">
            <h3>Generation</h3>
            <div style={{ marginBottom: '0.75rem' }}>
              <input
                type="text"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Prompt (optional) — leave empty for random start"
                style={{
                  width: '100%',
                  padding: '0.5rem 0.75rem',
                  background: '#222',
                  border: '1px solid #444',
                  borderRadius: '4px',
                  color: '#e0e0e0',
                  fontSize: '0.85rem',
                  fontFamily: "'SF Mono', 'Fira Code', 'Cascadia Code', monospace",
                  outline: 'none',
                  boxSizing: 'border-box',
                }}
              />
            </div>
            <div className="gen-sliders">
              <div className="gen-slider">
                <label>Temperature: {temperature.toFixed(1)}</label>
                <input
                  type="range"
                  min="0.1"
                  max="2.0"
                  step="0.1"
                  value={temperature}
                  onChange={(e) => setTemperature(parseFloat(e.target.value))}
                />
              </div>
              <div className="gen-slider">
                <label>Max Tokens: {maxTokens}</label>
                <input
                  type="range"
                  min="50"
                  max="500"
                  step="50"
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                />
              </div>
              <button
                className="btn-generate"
                onClick={handleGenerate}
                disabled={!trainer?.params || isTraining || generating}
              >
                {generating ? 'Generating...' : 'Generate'}
              </button>
            </div>
          </div>

          {sampleText && (
            <div className="sample-output">
              <h3>Generated Sample</h3>
              <pre className="sample-text">{sampleText}</pre>
            </div>
          )}
        </div>

        {/* Right column: architecture + checkpoints */}
        <div className="training-sidebar">
          <ArchitectureDiagram config={config} />

          <CheckpointPanel
            params={trainer?.params ?? null}
            config={trainer?.config ?? toLegacyConfig(config)}
            step={trainer?.step ?? 0}
            lossHistory={lossHistory}
            vocab={trainer?.tokenizer?.vocab ?? []}
            datasetId={dataset?.id ?? ''}
            onLoad={handleCheckpointLoad}
            disabled={isTraining}
          />
        </div>
      </div>
    </div>
  )
}
