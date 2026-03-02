import { encode as gpt2Encode, decode as gpt2Decode } from 'gpt-tokenizer/model/davinci'

// ---- Types ----

export type TokenizerType = 'char' | 'bpe-gpt2'

export interface TokenizerState {
  type: TokenizerType
  vocab?: string[] // only used by CharTokenizer
}

export interface Tokenizer {
  readonly type: TokenizerType
  readonly vocabSize: number
  init(text: string): void
  encode(text: string): Uint32Array
  decode(ids: Uint32Array | number[]): string
  getState(): TokenizerState
}

// ---- CharTokenizer ----

export class CharTokenizer implements Tokenizer {
  readonly type: TokenizerType = 'char'
  private charToIdx: Map<string, number> = new Map()
  private idxToChar: string[] = []

  get vocabSize(): number {
    return this.idxToChar.length
  }

  get vocab(): string[] {
    return [...this.idxToChar]
  }

  init(text: string): void {
    const chars = new Set(text)
    this.idxToChar = Array.from(chars).sort()
    this.charToIdx.clear()
    this.idxToChar.forEach((ch, i) => this.charToIdx.set(ch, i))
  }

  restoreVocab(vocab: string[]): void {
    this.idxToChar = [...vocab]
    this.charToIdx.clear()
    this.idxToChar.forEach((ch, i) => this.charToIdx.set(ch, i))
  }

  encode(text: string): Uint32Array {
    const codePoints = Array.from(text)
    const ids = new Uint32Array(codePoints.length)
    for (let i = 0; i < codePoints.length; i++) {
      const idx = this.charToIdx.get(codePoints[i])
      if (idx === undefined) throw new Error(`Unknown character: ${codePoints[i]}`)
      ids[i] = idx
    }
    return ids
  }

  decode(ids: Uint32Array | number[]): string {
    let result = ''
    for (const id of ids) {
      result += this.idxToChar[id] ?? '?'
    }
    return result
  }

  getState(): TokenizerState {
    return { type: 'char', vocab: [...this.idxToChar] }
  }
}

// ---- GPT2Tokenizer ----

const GPT2_VOCAB_SIZE = 50257

export class GPT2Tokenizer implements Tokenizer {
  readonly type: TokenizerType = 'bpe-gpt2'
  readonly vocabSize = GPT2_VOCAB_SIZE

  init(_text: string): void {
    // Pre-trained BPE — no corpus-dependent init needed
  }

  encode(text: string): Uint32Array {
    const ids = gpt2Encode(text)
    return new Uint32Array(ids)
  }

  decode(ids: Uint32Array | number[]): string {
    return gpt2Decode(Array.from(ids))
  }

  getState(): TokenizerState {
    return { type: 'bpe-gpt2' }
  }
}

// ---- Factory functions ----

export function createTokenizer(type: TokenizerType): Tokenizer {
  switch (type) {
    case 'bpe-gpt2':
      return new GPT2Tokenizer()
    case 'char':
      return new CharTokenizer()
    default:
      throw new Error(`Unknown tokenizer type: ${type}`)
  }
}

export function restoreTokenizer(state: TokenizerState, text?: string): Tokenizer {
  switch (state.type) {
    case 'char': {
      const tok = new CharTokenizer()
      if (state.vocab) {
        tok.restoreVocab(state.vocab)
      } else if (text) {
        tok.init(text)
      }
      return tok
    }
    case 'bpe-gpt2':
      return new GPT2Tokenizer()
    default:
      throw new Error(`Unknown tokenizer type: ${state.type}`)
  }
}
