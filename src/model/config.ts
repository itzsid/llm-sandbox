export interface TransformerConfig {
  vocabSize: number
  blockSize: number  // max sequence length
  nLayers: number
  nHeads: number
  dModel: number
  dFF: number
}

export const NANO_GPT_CONFIG: TransformerConfig = {
  vocabSize: 65,     // Tiny Shakespeare character vocab
  blockSize: 128,
  nLayers: 4,
  nHeads: 4,
  dModel: 128,
  dFF: 512,
}

export {
  type ModelConfig,
  type LayerConfig,
  type ConfigError,
  validateConfig,
  estimateParamCount,
  toLegacyConfig,
  fromLegacyConfig,
  configToText,
  textToConfig,
  PRESETS,
} from './schema'
