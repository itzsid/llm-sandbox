import type { ModelConfig } from '../model/schema'

export function encodeConfigToHash(config: ModelConfig): string {
  const json = JSON.stringify(config)
  const encoded = btoa(json)
  return `#config=${encoded}`
}

export function decodeConfigFromHash(hash: string): ModelConfig | null {
  try {
    const match = hash.match(/^#config=(.+)$/)
    if (!match) return null
    const json = atob(match[1])
    return JSON.parse(json) as ModelConfig
  } catch {
    return null
  }
}
