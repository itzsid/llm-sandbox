export class CharTokenizer {
  private charToIdx: Map<string, number> = new Map()
  private idxToChar: string[] = []

  get vocabSize(): number {
    return this.idxToChar.length
  }

  buildVocab(text: string): void {
    const chars = new Set(text)
    this.idxToChar = Array.from(chars).sort()
    this.charToIdx.clear()
    this.idxToChar.forEach((ch, i) => this.charToIdx.set(ch, i))
  }

  encode(text: string): Uint32Array {
    const ids = new Uint32Array(text.length)
    for (let i = 0; i < text.length; i++) {
      const idx = this.charToIdx.get(text[i])
      if (idx === undefined) throw new Error(`Unknown character: ${text[i]}`)
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
}
