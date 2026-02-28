import { Tensor } from '../gpu/tensor'
import { crossEntropy } from '../gpu/ops'
import { backward, clearTape, setGradEnabled } from '../gpu/autograd'
import { AdamOptimizer } from '../gpu/optimizer'
import { CharTokenizer } from './tokenizer'
import { initTransformer, transformerForward, getAllParams, type TransformerParams } from '../model/transformer'
import { type TransformerConfig, NANO_GPT_CONFIG } from '../model/config'
import tinyShakespeare from '../data/tiny-shakespeare.txt?raw'

export interface TrainingMetrics {
  step: number
  loss: number
  tokensPerSec: number
}

export class Trainer {
  private config: TransformerConfig
  private tokenizer: CharTokenizer | null = null
  private encodedText: Uint32Array | null = null
  private params: TransformerParams | null = null
  private optimizer: AdamOptimizer | null = null
  private allParams: Tensor[] = []
  private running = false
  private _step = 0
  private batchSize = 4
  private seqLen: number

  constructor(config?: TransformerConfig) {
    this.config = config ?? NANO_GPT_CONFIG
    this.seqLen = this.config.blockSize
  }

  async init(): Promise<void> {
    // Build tokenizer
    this.tokenizer = new CharTokenizer()
    this.tokenizer.buildVocab(tinyShakespeare)
    console.log(`Vocab size: ${this.tokenizer.vocabSize}`)

    // Override vocabSize from actual data
    this.config = { ...this.config, vocabSize: this.tokenizer.vocabSize }

    // Encode text
    this.encodedText = this.tokenizer.encode(tinyShakespeare)
    console.log(`Encoded ${this.encodedText.length} tokens`)

    // Init model
    this.params = await initTransformer(this.config)
    this.allParams = getAllParams(this.params)
    this.optimizer = new AdamOptimizer(this.allParams, { lr: 3e-4, weightDecay: 0.01 })

    const totalParams = this.allParams.reduce((sum, p) => sum + p.size, 0)
    console.log(`Model initialized: ${totalParams.toLocaleString()} parameters`)
  }

  sampleBatch(): { inputs: Uint32Array; targets: Uint32Array } {
    if (!this.encodedText) throw new Error('Not initialized')
    const B = this.batchSize
    const T = this.seqLen
    const inputs = new Uint32Array(B * T)
    const targets = new Uint32Array(B * T)
    const maxStart = this.encodedText.length - T - 1

    for (let b = 0; b < B; b++) {
      const start = Math.floor(Math.random() * maxStart)
      for (let t = 0; t < T; t++) {
        inputs[b * T + t] = this.encodedText[start + t]
        targets[b * T + t] = this.encodedText[start + t + 1]
      }
    }
    return { inputs, targets }
  }

  async train(
    onMetrics: (m: TrainingMetrics) => void,
    onSample: (text: string) => void,
  ): Promise<void> {
    if (!this.params || !this.optimizer) throw new Error('Not initialized')
    this.running = true

    while (this.running) {
      const stepStart = performance.now()

      // Sample batch
      const { inputs, targets } = this.sampleBatch()
      const inputTensor = await Tensor.fromU32(inputs, [this.batchSize, this.seqLen])
      const targetTensor = await Tensor.fromU32(targets, [this.batchSize * this.seqLen])

      // Forward
      setGradEnabled(true)
      const logits = await transformerForward(this.params, inputTensor, this.config)

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
      const elapsed = (performance.now() - stepStart) / 1000
      const tokensPerSec = (this.batchSize * this.seqLen) / elapsed

      onMetrics({
        step: this._step,
        loss: lossVal,
        tokensPerSec,
      })

      // Generate sample every 10 steps
      if (this._step % 10 === 0) {
        const sample = await this.generateSample(100)
        onSample(sample)
      }

      // Yield to UI
      await new Promise((r) => setTimeout(r, 0))
    }
  }

  async generateSample(maxTokens: number): Promise<string> {
    if (!this.params || !this.tokenizer) return ''

    setGradEnabled(false)
    try {
      // Start with a random character
      const startIdx = Math.floor(Math.random() * this.tokenizer.vocabSize)
      const generated: number[] = [startIdx]

      for (let i = 0; i < maxTokens; i++) {
        // Take last blockSize tokens
        const contextLen = Math.min(generated.length, this.config.blockSize)
        const context = generated.slice(-contextLen)
        const inputIds = new Uint32Array(context)
        const inputTensor = await Tensor.fromU32(inputIds, [1, contextLen])

        const logits = await transformerForward(this.params!, inputTensor, this.config)
        const logitsArr = await logits.toArray()

        // Get last token logits
        const lastOffset = (contextLen - 1) * this.config.vocabSize
        const lastLogits = logitsArr.slice(lastOffset, lastOffset + this.config.vocabSize)

        // Temperature sampling (temperature=0.8)
        const temperature = 0.8
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

      return this.tokenizer.decode(generated)
    } finally {
      setGradEnabled(true)
    }
  }

  stop(): void {
    this.running = false
  }

  get step(): number {
    return this._step
  }
}
