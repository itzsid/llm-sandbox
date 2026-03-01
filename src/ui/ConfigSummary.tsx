import type { ConfigError, ModelConfig } from '../model/schema'
import type { Dataset } from '../data/datasets'

export type TrainingStatus = 'idle' | 'initializing' | 'training' | 'stopped'

interface ConfigSummaryProps {
  config: ModelConfig
  paramCount: number
  errors: ConfigError[]
  dataset: Dataset | null
  trainingStatus: TrainingStatus
  onStart: () => void
  onStop: () => void
}

function formatParamCount(count: number): string {
  if (count >= 1_000_000) return `${(count / 1_000_000).toFixed(1)}M`
  if (count >= 1_000) return `${(count / 1_000).toFixed(0)}K`
  return String(count)
}

export function ConfigSummary({
  config,
  paramCount,
  errors,
  dataset,
  trainingStatus,
  onStart,
  onStop,
}: ConfigSummaryProps) {
  const hasErrors = errors.length > 0
  const isTraining = trainingStatus === 'training'
  const isInitializing = trainingStatus === 'initializing'
  const canStart = !hasErrors && !!dataset && !isTraining && !isInitializing

  const buttonLabel = isTraining
    ? 'Training...'
    : isInitializing
      ? 'Initializing...'
      : trainingStatus === 'stopped'
        ? 'Resume'
        : 'Start Training'

  const disabledReason = hasErrors
    ? 'Fix config errors first'
    : !dataset
      ? 'Select a dataset first'
      : isTraining
        ? 'Training in progress'
        : isInitializing
          ? 'Initializing model...'
          : undefined

  return (
    <div className="config-summary" style={isTraining ? { borderColor: 'var(--border-active)' } : undefined}>
      <div className="config-summary-items">
        <div className="config-summary-item">
          <span className="config-summary-label">Model</span>
          <span className="config-summary-value">{config.name}</span>
        </div>
        <div className="config-summary-item">
          <span className="config-summary-label">Params</span>
          <span className="config-summary-value">{formatParamCount(paramCount)}</span>
        </div>
        <div className="config-summary-item">
          <span className="config-summary-label">Layers</span>
          <span className="config-summary-value">{config.layers.length}</span>
        </div>
        <div className="config-summary-item">
          <span className="config-summary-label">Dataset</span>
          <span className="config-summary-value">
            {dataset ? dataset.name : <span style={{ color: 'var(--text-3)' }}>None</span>}
          </span>
        </div>
        {hasErrors && (
          <div className="config-summary-item config-summary-errors">
            <span className="config-summary-label">Errors</span>
            <span className="config-summary-value" style={{ color: 'var(--red)' }}>
              {errors.length} {errors.length === 1 ? 'issue' : 'issues'}
            </span>
          </div>
        )}
      </div>

      <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', flexShrink: 0 }}>
        {isTraining && (
          <button
            className="config-summary-stop"
            onClick={onStop}
          >
            Stop
          </button>
        )}
        <button
          className={`config-summary-train ${isTraining ? 'is-training' : ''}`}
          onClick={isTraining ? undefined : onStart}
          disabled={!canStart && !isTraining}
          title={disabledReason}
        >
          {isTraining && <span className="training-dot" />}
          {buttonLabel}
        </button>
      </div>
    </div>
  )
}
