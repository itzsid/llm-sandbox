import { Tensor } from '../gpu/tensor'
import { crossEntropy } from '../gpu/ops'
import { backward, clearTape, setGradEnabled } from '../gpu/autograd'
import { AdamOptimizer } from '../gpu/optimizer'
import { CharTokenizer } from './tokenizer'
import { initTransformer, transformerForward, getAllParams, deserializeParams, type TransformerParams } from '../model/transformer'
import { type TransformerConfig, NANO_GPT_CONFIG } from '../model/config'

export interface TrainingMetrics {
  step: number
  loss: number
  tokensPerSec: number
  valLoss?: number
  learningRate: number
  gradNorm?: number
}

export class Trainer {
  private _config: TransformerConfig
  private _tokenizer: CharTokenizer | null = null
  private encodedText: Uint32Array | null = null
  private valStart = 0
  private _params: TransformerParams | null = null
  private optimizer: AdamOptimizer | null = null
  private _allParams: Tensor[] = []
  private running = false
  private _step = 0
  private batchSize = 4
  private seqLen: number
  private lr = 3e-4
  private _lossHistory: number[] = []

  constructor(config?: TransformerConfig) {
    this._config = config ?? NANO_GPT_CONFIG
    this.seqLen = this._config.blockSize
  }

  get config(): TransformerConfig { return this._config }
  get params(): TransformerParams | null { return this._params }
  get tokenizer(): CharTokenizer | null { return this._tokenizer }
  get lossHistory(): number[] { return this._lossHistory }
  get step(): number { return this._step }
  get isRunning(): boolean { return this.running }

  async init(text: string): Promise<void> {
    this._tokenizer = new CharTokenizer()
    this._tokenizer.buildVocab(text)
    console.log(`Vocab size: ${this._tokenizer.vocabSize}`)

    this._config = { ...this._config, vocabSize: this._tokenizer.vocabSize }

    this.encodedText = this._tokenizer.encode(text)
    // 90/10 train/val split
    this.valStart = Math.floor(this.encodedText.length * 0.9)
    console.log(`Encoded ${this.encodedText.length} tokens (train: ${this.valStart}, val: ${this.encodedText.length - this.valStart})`)

    this._params = await initTransformer(this._config)
    this._allParams = getAllParams(this._params)
    this.optimizer = new AdamOptimizer(this._allParams, { lr: this.lr, weightDecay: 0.01 })

    const totalParams = this._allParams.reduce((sum, p) => sum + p.size, 0)
    console.log(`Model initialized: ${totalParams.toLocaleString()} parameters`)
  }

  async loadFromCheckpoint(
    serializedParams: Record<string, { shape: number[]; data: Float32Array }>,
    config: TransformerConfig,
    step: number,
    lossHistory: number[],
    _vocab: string[],
    text: string,
  ): Promise<void> {
    this._tokenizer = new CharTokenizer()
    this._tokenizer.buildVocab(text)
    this._config = config
    this.seqLen = config.blockSize
    this.encodedText = this._tokenizer.encode(text)
    this.valStart = Math.floor(this.encodedText.length * 0.9)

    this._params = await deserializeParams(serializedParams, config)
    this._allParams = getAllParams(this._params)
    this.optimizer = new AdamOptimizer(this._allParams, { lr: this.lr, weightDecay: 0.01 })
    this._step = step
    this._lossHistory = [...lossHistory]
  }

  private sampleBatch(validation = false): { inputs: Uint32Array; targets: Uint32Array } {
    if (!this.encodedText) throw new Error('Not initialized')
    const B = this.batchSize
    const T = this.seqLen
    const inputs = new Uint32Array(B * T)
    const targets = new Uint32Array(B * T)

    const start0 = validation ? this.valStart : 0
    const end0 = validation ? this.encodedText.length : this.valStart
    const rangeLen = end0 - start0 - T - 1

    for (let b = 0; b < B; b++) {
      const start = start0 + Math.floor(Math.random() * rangeLen)
      for (let t = 0; t < T; t++) {
        inputs[b * T + t] = this.encodedText[start + t]
        targets[b * T + t] = this.encodedText[start + t + 1]
      }
    }
    return { inputs, targets }
  }

  private async computeValLoss(): Promise<number> {
    if (!this._params) return 0
    setGradEnabled(false)
    try {
      const { inputs, targets } = this.sampleBatch(true)
      const inputTensor = await Tensor.fromU32(inputs, [this.batchSize, this.seqLen])
      const targetTensor = await Tensor.fromU32(targets, [this.batchSize * this.seqLen])
      const logits = await transformerForward(this._params, inputTensor, this._config)
      const loss = await crossEntropy(logits, targetTensor)
      const val = (await loss.toArray())[0]
      inputTensor.dispose()
      targetTensor.dispose()
      logits.dispose()
      loss.dispose()
      return val
    } finally {
      setGradEnabled(true)
    }
  }

  async train(
    onMetrics: (m: TrainingMetrics) => void,
    onSample: (text: string) => void,
  ): Promise<void> {
    if (!this._params || !this.optimizer) throw new Error('Not initialized')
    this.running = true

    while (this.running) {
      const stepStart = performance.now()

      // Sample batch
      const { inputs, targets } = this.sampleBatch(false)
      const inputTensor = await Tensor.fromU32(inputs, [this.batchSize, this.seqLen])
      const targetTensor = await Tensor.fromU32(targets, [this.batchSize * this.seqLen])

      // Forward
      setGradEnabled(true)
      const logits = await transformerForward(this._params, inputTensor, this._config)

      // Loss
      const loss = await crossEntropy(logits, targetTensor)
      const lossVal = (await loss.toArray())[0]

      // Backward
      await backward(loss)

      // Optimizer step
      await this.optimizer.step()

      // Cleanup
      this.optimizer.zeroGrad()
      clearTape()
      inputTensor.dispose()
      targetTensor.dispose()
      logits.dispose()
      loss.dispose()

      this._step++
      this._lossHistory.push(lossVal)

      const elapsed = (performance.now() - stepStart) / 1000
      const tokensPerSec = (this.batchSize * this.seqLen) / elapsed

      const metrics: TrainingMetrics = {
        step: this._step,
        loss: lossVal,
        tokensPerSec,
        learningRate: this.lr,
      }

      // Validation loss every 50 steps
      if (this._step % 50 === 0) {
        metrics.valLoss = await this.computeValLoss()
      }

      onMetrics(metrics)

      // Generate sample every 10 steps
      if (this._step % 10 === 0) {
        const sample = await this.generateSample(100)
        onSample(sample)
      }

      // Yield to UI
      await new Promise((r) => setTimeout(r, 0))
    }
  }

  async generateSample(maxTokens: number, temperature = 0.8): Promise<string> {
    if (!this._params || !this._tokenizer) return ''

    setGradEnabled(false)
    try {
      // Start with a random character
      const startIdx = Math.floor(Math.random() * this._tokenizer.vocabSize)
      const generated: number[] = [startIdx]

      for (let i = 0; i < maxTokens; i++) {
        // Take last blockSize tokens
        const contextLen = Math.min(generated.length, this._config.blockSize)
        const context = generated.slice(-contextLen)
        const inputIds = new Uint32Array(context)
        const inputTensor = await Tensor.fromU32(inputIds, [1, contextLen])

        const logits = await transformerForward(this._params!, inputTensor, this._config)
        const logitsArr = await logits.toArray()

        // Get last token logits
        const lastOffset = (contextLen - 1) * this._config.vocabSize
        const lastLogits = logitsArr.slice(lastOffset, lastOffset + this._config.vocabSize)

        // Temperature sampling
        const scaledLogits = lastLogits.map((l: number) => l / temperature)
        const maxLogit = Math.max(...scaledLogits)
        const exps = scaledLogits.map((l: number) => Math.exp(l - maxLogit))
        const sumExps = exps.reduce((a: number, b: number) => a + b, 0)
        const probs = exps.map((e: number) => e / sumExps)

        // Sample from distribution
        let r = Math.random()
        let nextToken = 0
        for (let j = 0; j < probs.length; j++) {
          r -= probs[j]
          if (r <= 0) {
            nextToken = j
            break
          }
        }
        generated.push(nextToken)

        inputTensor.dispose()
        logits.dispose()
      }

      return this._tokenizer.decode(generated)
    } finally {
      setGradEnabled(true)
    }
  }

  stop(): void {
    this.running = false
  }
}
