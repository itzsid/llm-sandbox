import { serializeParams, deserializeParams } from '../model/transformer'
import type { TransformerConfig } from '../model/config'
import type { TokenizerState } from '../training/tokenizer'
import type { TrainingHyperparams } from '../training/trainer'

export interface Checkpoint {
  version: number  // 3
  name: string
  config: TransformerConfig
  step: number
  lossHistory: number[]
  params: Record<string, { shape: number[]; data: Float32Array }>
  tokenizer: TokenizerState
  datasetId: string
  savedAt: number
  hyperparams?: TrainingHyperparams
  valLossHistory?: number[]
}

const DB_NAME = 'llm-sandbox'
const STORE_NAME = 'checkpoints'
const DB_VERSION = 1

function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION)
    request.onupgradeneeded = () => {
      const db = request.result
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: 'name' })
      }
    }
    request.onsuccess = () => resolve(request.result)
    request.onerror = () => reject(request.error)
    request.onblocked = () => reject(new Error('IndexedDB open blocked — close other tabs using this app'))
  })
}

/**
 * Convert a Checkpoint to a storable format where Float32Arrays
 * become regular arrays (IndexedDB structured clone handles them,
 * but we do it explicitly for safety).
 */
function toStorable(checkpoint: Checkpoint): unknown {
  const params: Record<string, { shape: number[]; data: number[] }> = {}
  for (const [key, val] of Object.entries(checkpoint.params)) {
    params[key] = {
      shape: val.shape,
      data: Array.from(val.data),
    }
  }
  return {
    ...checkpoint,
    params,
  }
}

/**
 * Convert from stored format back to Checkpoint with Float32Arrays.
 */
function fromStorable(stored: any): Checkpoint {
  const params: Record<string, { shape: number[]; data: Float32Array }> = {}
  for (const [key, val] of Object.entries(stored.params as Record<string, { shape: number[]; data: number[] }>)) {
    params[key] = {
      shape: val.shape,
      data: new Float32Array(val.data),
    }
  }

  // v1 → v2 migration: convert vocab array to TokenizerState
  let tokenizer: TokenizerState
  if (stored.tokenizer) {
    tokenizer = stored.tokenizer
  } else if (stored.vocab) {
    tokenizer = { type: 'char', vocab: stored.vocab }
  } else {
    tokenizer = { type: 'bpe-gpt2' }
  }

  return {
    version: 3,
    name: stored.name,
    config: stored.config,
    step: stored.step,
    lossHistory: stored.lossHistory,
    params,
    tokenizer,
    datasetId: stored.datasetId,
    savedAt: stored.savedAt,
    hyperparams: stored.hyperparams,
    valLossHistory: stored.valLossHistory,
  }
}

export async function saveCheckpoint(checkpoint: Checkpoint): Promise<void> {
  const db = await openDB()
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readwrite')
    const store = tx.objectStore(STORE_NAME)
    store.put(toStorable(checkpoint))
    tx.oncomplete = () => { db.close(); resolve() }
    tx.onerror = () => { db.close(); reject(tx.error) }
    tx.onabort = () => { db.close(); reject(tx.error ?? new Error('Transaction aborted')) }
  })
}

export async function loadCheckpoint(name: string): Promise<Checkpoint> {
  const db = await openDB()
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readonly')
    const store = tx.objectStore(STORE_NAME)
    const request = store.get(name)
    let result: any = undefined
    request.onsuccess = () => { result = request.result }
    tx.oncomplete = () => {
      db.close()
      if (!result) {
        reject(new Error(`Checkpoint "${name}" not found`))
      } else {
        resolve(fromStorable(result))
      }
    }
    tx.onerror = () => { db.close(); reject(tx.error) }
    tx.onabort = () => { db.close(); reject(tx.error ?? new Error('Transaction aborted')) }
  })
}

export async function listCheckpoints(): Promise<{ name: string; step: number; savedAt: number; datasetId: string }[]> {
  const db = await openDB()
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readonly')
    const store = tx.objectStore(STORE_NAME)
    const request = store.getAll()
    let rawResults: any[] = []
    request.onsuccess = () => { rawResults = request.result || [] }
    tx.oncomplete = () => {
      db.close()
      const results = rawResults.map((item: any) => ({
        name: item.name as string,
        step: item.step as number,
        savedAt: item.savedAt as number,
        datasetId: item.datasetId as string,
      }))
      // Sort by savedAt descending (most recent first)
      results.sort((a, b) => b.savedAt - a.savedAt)
      resolve(results)
    }
    tx.onerror = () => { db.close(); reject(tx.error) }
    tx.onabort = () => { db.close(); reject(tx.error ?? new Error('Transaction aborted')) }
  })
}

export async function deleteCheckpoint(name: string): Promise<void> {
  const db = await openDB()
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readwrite')
    const store = tx.objectStore(STORE_NAME)
    store.delete(name)
    tx.oncomplete = () => { db.close(); resolve() }
    tx.onerror = () => { db.close(); reject(tx.error) }
    tx.onabort = () => { db.close(); reject(tx.error ?? new Error('Transaction aborted')) }
  })
}

// ---- File Export/Import ----

/**
 * Binary format:
 * [4-byte header length (uint32 LE)]
 * [JSON metadata: { version, name, config, step, lossHistory, vocab, datasetId, savedAt, paramEntries: [{key, shape, offset, length}] }]
 * [concatenated Float32Array data for all params]
 */
export function exportCheckpoint(checkpoint: Checkpoint): Blob {
  // Build the param data buffer and index
  const paramEntries: { key: string; shape: number[]; offset: number; length: number }[] = []
  const dataChunks: Float32Array[] = []
  let offset = 0

  for (const [key, val] of Object.entries(checkpoint.params)) {
    const byteLength = val.data.byteLength
    paramEntries.push({
      key,
      shape: val.shape,
      offset,
      length: val.data.length,
    })
    dataChunks.push(val.data)
    offset += byteLength
  }

  const metadata = JSON.stringify({
    version: 3,
    name: checkpoint.name,
    config: checkpoint.config,
    step: checkpoint.step,
    lossHistory: checkpoint.lossHistory,
    tokenizer: checkpoint.tokenizer,
    datasetId: checkpoint.datasetId,
    savedAt: checkpoint.savedAt,
    hyperparams: checkpoint.hyperparams,
    valLossHistory: checkpoint.valLossHistory,
    paramEntries,
  })

  const metadataBytes = new TextEncoder().encode(metadata)
  const headerLen = metadataBytes.byteLength

  // Total size: 4 (header length) + headerLen + total param data
  const totalParamBytes = offset
  const totalSize = 4 + headerLen + totalParamBytes
  const buffer = new ArrayBuffer(totalSize)
  const view = new DataView(buffer)

  // Write header length (uint32 LE)
  view.setUint32(0, headerLen, true)

  // Write metadata
  new Uint8Array(buffer, 4, headerLen).set(metadataBytes)

  // Write param data
  let writeOffset = 4 + headerLen
  for (const chunk of dataChunks) {
    new Uint8Array(buffer, writeOffset, chunk.byteLength).set(new Uint8Array(chunk.buffer, chunk.byteOffset, chunk.byteLength))
    writeOffset += chunk.byteLength
  }

  return new Blob([buffer], { type: 'application/octet-stream' })
}

export function importCheckpointFromBuffer(arrayBuffer: ArrayBuffer): Checkpoint {
  const view = new DataView(arrayBuffer)

  // Read header length
  const headerLen = view.getUint32(0, true)

  // Read metadata
  const metadataBytes = new Uint8Array(arrayBuffer, 4, headerLen)
  const metadata = JSON.parse(new TextDecoder().decode(metadataBytes))

  // Read param data
  const paramDataStart = 4 + headerLen
  const params: Record<string, { shape: number[]; data: Float32Array }> = {}

  for (const entry of metadata.paramEntries as { key: string; shape: number[]; offset: number; length: number }[]) {
    const byteOffset = paramDataStart + entry.offset
    // Copy data to avoid issues with shared buffer
    const data = new Float32Array(entry.length)
    const source = new Float32Array(arrayBuffer, byteOffset, entry.length)
    data.set(source)
    params[entry.key] = {
      shape: entry.shape,
      data,
    }
  }

  // v1 → v2 migration for imported files
  let tokenizer: TokenizerState
  if (metadata.tokenizer) {
    tokenizer = metadata.tokenizer
  } else if (metadata.vocab) {
    tokenizer = { type: 'char', vocab: metadata.vocab }
  } else {
    tokenizer = { type: 'bpe-gpt2' }
  }

  return {
    version: 3,
    name: metadata.name,
    config: metadata.config,
    step: metadata.step,
    lossHistory: metadata.lossHistory,
    params,
    tokenizer,
    datasetId: metadata.datasetId,
    savedAt: metadata.savedAt,
    hyperparams: metadata.hyperparams,
    valLossHistory: metadata.valLossHistory,
  }
}

export async function importCheckpoint(file: File): Promise<Checkpoint> {
  const arrayBuffer = await file.arrayBuffer()
  return importCheckpointFromBuffer(arrayBuffer)
}

export { serializeParams, deserializeParams }
