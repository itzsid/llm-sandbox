export interface HighlightToken {
  text: string
  type: 'string' | 'number' | 'property' | 'punctuation' | 'keyword' | 'comment' | 'whitespace'
}

/**
 * Tokenize JSON text (with // comment support) into highlight tokens.
 */
export function tokenizeJSON(text: string): HighlightToken[] {
  const tokens: HighlightToken[] = []
  let i = 0

  while (i < text.length) {
    const ch = text[i]

    // Whitespace
    if (ch === ' ' || ch === '\t' || ch === '\n' || ch === '\r') {
      let start = i
      while (i < text.length && (text[i] === ' ' || text[i] === '\t' || text[i] === '\n' || text[i] === '\r')) {
        i++
      }
      tokens.push({ text: text.slice(start, i), type: 'whitespace' })
      continue
    }

    // Single-line comment
    if (ch === '/' && i + 1 < text.length && text[i + 1] === '/') {
      let start = i
      while (i < text.length && text[i] !== '\n') {
        i++
      }
      tokens.push({ text: text.slice(start, i), type: 'comment' })
      continue
    }

    // String (double-quoted)
    if (ch === '"') {
      let start = i
      i++ // skip opening quote
      while (i < text.length && text[i] !== '"') {
        if (text[i] === '\\') {
          i++ // skip escaped character
        }
        i++
      }
      if (i < text.length) {
        i++ // skip closing quote
      }
      const str = text.slice(start, i)

      // Look ahead to determine if this is a property name (followed by colon)
      let j = i
      while (j < text.length && (text[j] === ' ' || text[j] === '\t')) {
        j++
      }
      if (j < text.length && text[j] === ':') {
        tokens.push({ text: str, type: 'property' })
      } else {
        tokens.push({ text: str, type: 'string' })
      }
      continue
    }

    // Number (integer, float, negative)
    if (ch === '-' || (ch >= '0' && ch <= '9')) {
      let start = i
      if (ch === '-') i++
      // Integer part
      while (i < text.length && text[i] >= '0' && text[i] <= '9') {
        i++
      }
      // Decimal part
      if (i < text.length && text[i] === '.') {
        i++
        while (i < text.length && text[i] >= '0' && text[i] <= '9') {
          i++
        }
      }
      // Exponent part
      if (i < text.length && (text[i] === 'e' || text[i] === 'E')) {
        i++
        if (i < text.length && (text[i] === '+' || text[i] === '-')) {
          i++
        }
        while (i < text.length && text[i] >= '0' && text[i] <= '9') {
          i++
        }
      }
      const numText = text.slice(start, i)
      // Only treat as number if we actually parsed digits (not just a bare '-')
      if (numText !== '-') {
        tokens.push({ text: numText, type: 'number' })
      } else {
        tokens.push({ text: numText, type: 'punctuation' })
      }
      continue
    }

    // Keywords: true, false, null
    if (text.slice(i, i + 4) === 'true' && !isAlphaNum(text[i + 4])) {
      tokens.push({ text: 'true', type: 'keyword' })
      i += 4
      continue
    }
    if (text.slice(i, i + 5) === 'false' && !isAlphaNum(text[i + 5])) {
      tokens.push({ text: 'false', type: 'keyword' })
      i += 5
      continue
    }
    if (text.slice(i, i + 4) === 'null' && !isAlphaNum(text[i + 4])) {
      tokens.push({ text: 'null', type: 'keyword' })
      i += 4
      continue
    }

    // Punctuation
    if (ch === '{' || ch === '}' || ch === '[' || ch === ']' || ch === ':' || ch === ',') {
      tokens.push({ text: ch, type: 'punctuation' })
      i++
      continue
    }

    // Anything else: consume one character as punctuation
    tokens.push({ text: ch, type: 'punctuation' })
    i++
  }

  return tokens
}

function isAlphaNum(ch: string | undefined): boolean {
  if (ch === undefined) return false
  return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') || ch === '_'
}

const COLOR_MAP: Record<HighlightToken['type'], string | null> = {
  property: '#93C5FD',
  string: '#BDE0A1',
  number: '#F59E0B',
  keyword: '#C4B5FD',
  punctuation: '#94A3B8',
  comment: '#4B5563',
  whitespace: null,
}

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
}

/**
 * Convert JSON text to HTML with colored spans for syntax highlighting.
 */
export function highlightJSON(text: string): string {
  const tokens = tokenizeJSON(text)
  let html = ''

  for (const token of tokens) {
    const escaped = escapeHtml(token.text)
    const color = COLOR_MAP[token.type]
    if (color) {
      html += `<span style="color:${color}">${escaped}</span>`
    } else {
      html += escaped
    }
  }

  return html
}
