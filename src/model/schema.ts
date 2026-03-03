import type { TransformerConfig } from './config'
import type { TokenizerType } from '../training/tokenizer'

// ---- Interfaces ----

export interface ModelConfig {
  name: string
  vocabSize: number | 'auto'  // 'auto' means derived from dataset at training time
  blockSize: number           // context window size
  layers: LayerConfig[]
  tieWeights: boolean         // tie token embedding and LM head weights
  tokenizerType?: TokenizerType
}

export interface LayerConfig {
  type: 'transformer_block'
  dModel: number
  nHeads: number
  dFF: number
  activation: 'gelu' | 'relu'
  normType: 'pre'
  dropout: number  // 0 for now (no dropout in WebGPU yet), but configurable
}

export interface ConfigError {
  path: string    // e.g. "blockSize", "layers[0].dModel"
  message: string
  suggestion?: string
  severity?: 'error' | 'warning'  // defaults to 'error'
}

// ---- Validation ----

export function validateConfig(config: ModelConfig): ConfigError[] {
  const errors: ConfigError[] = []

  // Name must be non-empty
  if (!config.name || config.name.trim().length === 0) {
    errors.push({ path: 'name', message: 'Name must be non-empty' })
  }

  // vocabSize: must be 'auto' or a positive integer
  if (config.vocabSize !== 'auto') {
    if (typeof config.vocabSize !== 'number' || !Number.isInteger(config.vocabSize) || config.vocabSize < 1) {
      errors.push({ path: 'vocabSize', message: 'vocabSize must be a positive integer or "auto"' })
    }
  }

  // blockSize: 8-1024
  if (typeof config.blockSize !== 'number' || !Number.isInteger(config.blockSize)) {
    errors.push({ path: 'blockSize', message: 'blockSize must be an integer', suggestion: 'Try 64, 128, or 256' })
  } else if (config.blockSize < 8 || config.blockSize > 1024) {
    errors.push({ path: 'blockSize', message: 'blockSize must be between 8 and 1024', suggestion: 'Common values: 64, 128, 256, 512' })
  }

  // layers: 1-12
  if (!Array.isArray(config.layers)) {
    errors.push({ path: 'layers', message: 'layers must be an array' })
    return errors // can't validate further
  }

  if (config.layers.length < 1 || config.layers.length > 12) {
    errors.push({ path: 'layers', message: 'Must have between 1 and 12 layers', suggestion: 'Start with 2-4 layers for small models' })
  }

  // All layers must have the same dModel
  const dModelValues = new Set(config.layers.map((l) => l.dModel))
  if (dModelValues.size > 1) {
    errors.push({
      path: 'layers',
      message: 'All layers must have the same dModel (transformer constraint)',
    })
  }

  // All layers must have the same nHeads
  if (config.layers.length > 1) {
    const nHeads0 = config.layers[0].nHeads
    for (let i = 1; i < config.layers.length; i++) {
      if (config.layers[i].nHeads !== nHeads0) {
        errors.push({
          path: 'layers',
          message: `All layers must have the same nHeads (layer 0 has ${nHeads0}, layer ${i} has ${config.layers[i].nHeads})`,
        })
        break
      }
    }
  }

  // All layers must have the same dFF
  if (config.layers.length > 1) {
    const dFF0 = config.layers[0].dFF
    for (let i = 1; i < config.layers.length; i++) {
      if (config.layers[i].dFF !== dFF0) {
        errors.push({
          path: 'layers',
          message: `All layers must have the same dFF (layer 0 has ${dFF0}, layer ${i} has ${config.layers[i].dFF})`,
        })
        break
      }
    }
  }

  // Warn about GPT-2 vocab with small models
  if (config.vocabSize === 'auto' && config.tokenizerType === 'bpe-gpt2' && config.layers.length > 0) {
    const dModel = config.layers[0].dModel
    if (dModel <= 128) {
      const embeddingParams = 50257 * dModel
      const totalEstimate = estimateParamCount(config)
      if (embeddingParams > totalEstimate * 0.5) {
        errors.push({
          path: 'tokenizerType',
          message: `GPT-2 tokenizer (50257 vocab) creates a ${(embeddingParams / 1e6).toFixed(1)}M param embedding — over half your ${(totalEstimate / 1e6).toFixed(1)}M model`,
          suggestion: 'Consider using "char" tokenizer or setting a smaller explicit vocabSize for tiny models',
          severity: 'warning',
        })
      }
    }
  }

  // Per-layer validation
  config.layers.forEach((layer, i) => {
    const prefix = `layers[${i}]`

    // dModel: 16-512
    if (typeof layer.dModel !== 'number' || !Number.isInteger(layer.dModel)) {
      errors.push({ path: `${prefix}.dModel`, message: 'dModel must be an integer', suggestion: 'Try 64, 128, or 256' })
    } else if (layer.dModel < 16 || layer.dModel > 512) {
      errors.push({ path: `${prefix}.dModel`, message: 'dModel must be between 16 and 512', suggestion: 'Common values: 64, 128, 256' })
    }

    // nHeads: 1-16
    if (typeof layer.nHeads !== 'number' || !Number.isInteger(layer.nHeads)) {
      errors.push({ path: `${prefix}.nHeads`, message: 'nHeads must be an integer', suggestion: 'Try 2, 4, or 8' })
    } else if (layer.nHeads < 1 || layer.nHeads > 16) {
      errors.push({ path: `${prefix}.nHeads`, message: 'nHeads must be between 1 and 16', suggestion: 'Common values: 2, 4, 8' })
    }

    // dFF: 16-2048
    if (typeof layer.dFF !== 'number' || !Number.isInteger(layer.dFF)) {
      errors.push({ path: `${prefix}.dFF`, message: 'dFF must be an integer', suggestion: 'Typically 4x dModel (e.g. 512 for dModel=128)' })
    } else if (layer.dFF < 16 || layer.dFF > 2048) {
      errors.push({ path: `${prefix}.dFF`, message: 'dFF must be between 16 and 2048', suggestion: 'Typically 4x dModel' })
    }

    // dModel must be divisible by nHeads
    if (
      typeof layer.dModel === 'number' &&
      typeof layer.nHeads === 'number' &&
      Number.isInteger(layer.dModel) &&
      Number.isInteger(layer.nHeads) &&
      layer.nHeads > 0 &&
      layer.dModel % layer.nHeads !== 0
    ) {
      // Suggest valid nHeads values
      const validHeads = []
      for (let h = 1; h <= Math.min(16, layer.dModel); h++) {
        if (layer.dModel % h === 0) validHeads.push(h)
      }
      errors.push({
        path: `${prefix}.nHeads`,
        message: `dModel (${layer.dModel}) must be divisible by nHeads (${layer.nHeads})`,
        suggestion: `Valid nHeads for dModel=${layer.dModel}: ${validHeads.join(', ')}`,
      })
    }

    // dropout: must be a number >= 0
    if (typeof layer.dropout !== 'number' || layer.dropout < 0 || layer.dropout > 1) {
      errors.push({ path: `${prefix}.dropout`, message: 'dropout must be between 0 and 1' })
    }

    // activation must be gelu or relu
    if (layer.activation !== 'gelu' && layer.activation !== 'relu') {
      errors.push({ path: `${prefix}.activation`, message: 'activation must be "gelu" or "relu"' })
    }

    // normType must be 'pre'
    if (layer.normType !== 'pre') {
      errors.push({ path: `${prefix}.normType`, message: 'normType must be "pre"' })
    }

    // type must be 'transformer_block'
    if (layer.type !== 'transformer_block') {
      errors.push({ path: `${prefix}.type`, message: 'type must be "transformer_block"' })
    }
  })

  return errors
}

// ---- Parameter estimation ----

export function estimateParamCount(config: ModelConfig): number {
  const vocabSize = config.vocabSize === 'auto'
    ? (config.tokenizerType === 'char' ? 65 : 50257)
    : config.vocabSize
  const nLayers = config.layers.length
  if (nLayers === 0) return 0

  const dModel = config.layers[0].dModel

  // Token embedding: vocabSize * dModel
  let params = vocabSize * dModel

  // Positional embedding: blockSize * dModel
  params += config.blockSize * dModel

  // Per-layer parameters
  for (const layer of config.layers) {
    const d = layer.dModel
    const ff = layer.dFF

    // Self-attention: Wq, Wk, Wv, Wo — each is dModel x dModel, plus biases
    // 4 * dModel * dModel (weights) + 4 * dModel (biases)
    params += 4 * d * d + 4 * d

    // Layer norm 1 (pre-attention): 2 * dModel (scale + bias)
    params += 2 * d

    // FFN: dModel -> dFF (weight + bias) + dFF -> dModel (weight + bias)
    params += d * ff + ff    // first linear
    params += ff * d + d     // second linear

    // Layer norm 2 (pre-FFN): 2 * dModel (scale + bias)
    params += 2 * d
  }

  // Final layer norm: 2 * dModel
  params += 2 * dModel

  // LM head: dModel * vocabSize (+ bias vocabSize)
  // If tieWeights, the LM head weight matrix is shared with token embedding
  if (!config.tieWeights) {
    params += dModel * vocabSize
  }

  return params
}

// ---- Conversion functions ----

export function toLegacyConfig(config: ModelConfig, actualVocabSize?: number): TransformerConfig {
  const layer = config.layers[0]
  const vocabSize =
    actualVocabSize != null
      ? actualVocabSize
      : config.vocabSize === 'auto'
        ? (config.tokenizerType === 'char' ? 65 : 50257)
        : config.vocabSize

  return {
    vocabSize,
    blockSize: config.blockSize,
    nLayers: config.layers.length,
    nHeads: layer.nHeads,
    dModel: layer.dModel,
    dFF: layer.dFF,
    tieWeights: config.tieWeights,
  }
}

export function fromLegacyConfig(config: TransformerConfig): ModelConfig {
  const layers: LayerConfig[] = Array(config.nLayers)
    .fill(null)
    .map(() => ({
      type: 'transformer_block' as const,
      dModel: config.dModel,
      nHeads: config.nHeads,
      dFF: config.dFF,
      activation: 'gelu' as const,
      normType: 'pre' as const,
      dropout: 0,
    }))

  return {
    name: 'Imported',
    vocabSize: config.vocabSize,
    blockSize: config.blockSize,
    layers,
    tieWeights: false,
  }
}

// ---- Serialization ----

export function configToText(config: ModelConfig): string {
  return JSON.stringify(config, null, 2)
}

export function textToConfig(text: string): ModelConfig {
  // Strip single-line comments (// ...)
  const stripped = text.replace(/\/\/.*$/gm, '')
  return JSON.parse(stripped) as ModelConfig
}

// ---- Presets ----

function makeLayerConfig(dModel: number, nHeads: number, dFF: number): LayerConfig {
  return {
    type: 'transformer_block',
    dModel,
    nHeads,
    dFF,
    activation: 'gelu',
    normType: 'pre',
    dropout: 0,
  }
}

export const PRESETS: Record<string, ModelConfig> = {
  tiny: {
    name: 'Tiny',
    vocabSize: 'auto',
    blockSize: 64,
    layers: Array(2)
      .fill(null)
      .map(() => makeLayerConfig(64, 4, 256)),
    tieWeights: false,
    tokenizerType: 'char',
  },
  nano: {
    name: 'Nano',
    vocabSize: 'auto',
    blockSize: 128,
    layers: Array(4)
      .fill(null)
      .map(() => makeLayerConfig(128, 4, 512)),
    tieWeights: false,
    tokenizerType: 'char',
  },
  micro: {
    name: 'Micro',
    vocabSize: 'auto',
    blockSize: 256,
    layers: Array(6)
      .fill(null)
      .map(() => makeLayerConfig(256, 8, 1024)),
    tieWeights: false,
    tokenizerType: 'char',
  },
}
