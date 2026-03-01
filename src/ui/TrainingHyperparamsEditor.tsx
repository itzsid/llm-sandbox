import type { TrainingHyperparams } from '../training/trainer'

interface TrainingHyperparamsEditorProps {
  hyperparams: TrainingHyperparams
  onChange: (h: TrainingHyperparams) => void
  disabled?: boolean
}

export function TrainingHyperparamsEditor({ hyperparams, onChange, disabled }: TrainingHyperparamsEditorProps) {
  return (
    <div className="hp-form">
      <div className="hp-grid">
        <HpField label="Max Steps" hint="10% warmup, 20% decay">
          <input
            type="number"
            className="hp-input"
            value={hyperparams.maxSteps}
            min={100}
            max={100000}
            step={100}
            onChange={(e) => onChange({ ...hyperparams, maxSteps: parseInt(e.target.value) || 100 })}
            disabled={disabled}
          />
        </HpField>
        <HpField label="Peak LR" hint="base learning rate">
          <input
            type="number"
            className="hp-input"
            value={hyperparams.lr}
            min={0.00001}
            max={0.01}
            step={0.0001}
            onChange={(e) => onChange({ ...hyperparams, lr: parseFloat(e.target.value) || hyperparams.lr })}
            disabled={disabled}
          />
        </HpField>
        <HpField label="Min LR" hint="floor after decay">
          <input
            type="number"
            className="hp-input"
            value={hyperparams.minLR}
            min={0}
            max={0.001}
            step={0.00001}
            onChange={(e) => onChange({ ...hyperparams, minLR: parseFloat(e.target.value) || 0 })}
            disabled={disabled}
          />
        </HpField>
        <HpField label="Weight Decay" hint="L2 regularization">
          <input
            type="number"
            className="hp-input"
            value={hyperparams.weightDecay}
            min={0}
            max={1}
            step={0.01}
            onChange={(e) => onChange({ ...hyperparams, weightDecay: parseFloat(e.target.value) || 0 })}
            disabled={disabled}
          />
        </HpField>
        <HpField label="Batch Size" hint="sequences per step">
          <input
            type="number"
            className="hp-input"
            value={hyperparams.batchSize}
            min={1}
            max={32}
            step={1}
            onChange={(e) => onChange({ ...hyperparams, batchSize: parseInt(e.target.value) || 1 })}
            disabled={disabled}
          />
        </HpField>
      </div>
      <div className="hp-schedule-summary">
        Warmup: {Math.floor(hyperparams.maxSteps * 0.1).toLocaleString()}
        &nbsp;&middot;&nbsp;
        Stable: {(Math.floor(hyperparams.maxSteps * 0.8) - Math.floor(hyperparams.maxSteps * 0.1)).toLocaleString()}
        &nbsp;&middot;&nbsp;
        Decay: {(hyperparams.maxSteps - Math.floor(hyperparams.maxSteps * 0.8)).toLocaleString()} steps
      </div>
    </div>
  )
}

function HpField({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <div className="hp-field">
      <label className="hp-label">
        {label}
        {hint && <span className="hp-hint"> ({hint})</span>}
      </label>
      {children}
    </div>
  )
}
