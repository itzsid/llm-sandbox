import { useState, useRef, useCallback, useEffect } from 'react'
import { Trainer, type TrainingMetrics } from '../training/trainer'
import { MetricsChart } from './MetricsChart'
import { CheckpointPanel } from './CheckpointPanel'
import { ArchitectureDiagram } from './ArchitectureDiagram'
import { toLegacyConfig } from '../model/schema'
import type { ModelConfig } from '../model/schema'
import type { Dataset } from '../data/datasets'
import type { Checkpoint } from '../storage/checkpoint'

interface TrainingPanelProps {
  config: ModelConfig
  dataset: Dataset | null
  onTrainingStateChange: (active: boolean) => void
}

export function TrainingPanel({ config, dataset, onTrainingStateChange }: TrainingPanelProps) {
  const [status, setStatus] = useState<'idle' | 'initializing' | 'training' | 'stopped'>('idle')
  const [metrics, setMetrics] = useState<TrainingMetrics | null>(null)
  const [lossHistory, setLossHistory] = useState<number[]>([])
  const [valLossHistory, setValLossHistory] = useState<number[]>([])
  const [tokensPerSecHistory, setTokensPerSecHistory] = useState<number[]>([])
  const [sampleText, setSampleText] = useState<string>('')
  const [error, setError] = useState<string | null>(null)
  const [temperature, setTemperature] = useState(0.8)
  const [maxTokens, setMaxTokens] = useState(100)
  const [generating, setGenerating] = useState(false)
  const trainerRef = useRef<Trainer | null>(null)

  const isTraining = status === 'training'

  useEffect(() => {
    onTrainingStateChange(isTraining)
  }, [isTraining, onTrainingStateChange])

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
      trainerRef.current = trainer
      await trainer.init(dataset.text)

      setStatus('training')
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
      const text = await trainer.generateSample(maxTokens, temperature)
      setSampleText(text)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setGenerating(false)
    }
  }, [temperature, maxTokens])

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
      trainerRef.current = trainer
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
      <div className="training-layout">
        {/* Left column: controls + metrics */}
        <div className="training-main">
          <div className="controls">
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
          </div>

          {error && <div className="error-msg">{error}</div>}

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
            </div>
          )}

          {lossHistory.length > 1 && (
            <MetricsChart
              data={lossHistory}
              label="Training Loss"
              color="#4caf50"
              height={140}
              formatValue={(v) => v.toFixed(2)}
            />
          )}

          {valLossHistory.length > 1 && (
            <MetricsChart
              data={valLossHistory}
              label="Validation Loss"
              color="#2196f3"
              height={120}
              formatValue={(v) => v.toFixed(2)}
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

          {/* Generation controls */}
          <div className="gen-controls">
            <h3>Generation</h3>
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
