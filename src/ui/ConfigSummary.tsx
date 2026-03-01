import type { ConfigError, ModelConfig } from '../model/schema'
import type { Dataset } from '../data/datasets'

interface ConfigSummaryProps {
  config: ModelConfig
  paramCount: number
  errors: ConfigError[]
  dataset: Dataset | null
  trainingActive: boolean
  onStartTraining: () => void
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
  trainingActive,
  onStartTraining,
}: ConfigSummaryProps) {
  const hasErrors = errors.length > 0
  const canStart = !hasErrors && dataset && !trainingActive

  return (
    <div className="config-summary">
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
            {dataset ? dataset.name : <span style={{ color: '#888' }}>None</span>}
          </span>
        </div>
        {hasErrors && (
          <div className="config-summary-item config-summary-errors">
            <span className="config-summary-label">Errors</span>
            <span className="config-summary-value" style={{ color: '#ff8a80' }}>
              {errors.length} {errors.length === 1 ? 'issue' : 'issues'}
            </span>
          </div>
        )}
      </div>
      <button
        className={`btn ${canStart ? 'btn-primary' : 'btn-secondary'}`}
        onClick={onStartTraining}
        disabled={!canStart}
        title={
          hasErrors
            ? 'Fix config errors first'
            : !dataset
              ? 'Select a dataset first'
              : trainingActive
                ? 'Training in progress'
                : 'Start training'
        }
      >
        {trainingActive ? 'Training...' : 'Start Training'}
      </button>
    </div>
  )
}
