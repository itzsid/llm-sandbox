export interface Dataset {
  id: string
  name: string
  description: string
  size: string
  text: string
}

type DatasetMeta = Omit<Dataset, 'text'>

const BUILTIN_DATASETS: DatasetMeta[] = [
  {
    id: 'tiny-shakespeare',
    name: 'Tiny Shakespeare',
    description: 'Complete works of Shakespeare (~1.1MB). Character-level language modeling on Early Modern English prose, poetry, and dialogue.',
    size: '1.1MB',
  },
  {
    id: 'tiny-chat',
    name: 'Tiny Chat',
    description: 'Small chat-format corpus with <user> and <assistant> tags. Covers greetings, weather, food, programming, math, science, and more.',
    size: '~5KB',
  },
]

export function getBuiltinDatasets(): DatasetMeta[] {
  return BUILTIN_DATASETS
}

export async function loadBuiltinDataset(id: string): Promise<Dataset> {
  const meta = BUILTIN_DATASETS.find((d) => d.id === id)
  if (!meta) {
    throw new Error(`Unknown built-in dataset: ${id}`)
  }

  let text: string
  switch (id) {
    case 'tiny-shakespeare': {
      const mod = await import('./tiny-shakespeare.txt?raw')
      text = mod.default
      break
    }
    case 'tiny-chat': {
      const mod = await import('./tiny-chat.txt?raw')
      text = mod.default
      break
    }
    default:
      throw new Error(`Unknown built-in dataset: ${id}`)
  }

  return { ...meta, text }
}

export function createCustomDataset(name: string, text: string): Dataset {
  const sizeBytes = new Blob([text]).size
  let size: string
  if (sizeBytes >= 1024 * 1024) {
    size = `${(sizeBytes / (1024 * 1024)).toFixed(1)}MB`
  } else if (sizeBytes >= 1024) {
    size = `${(sizeBytes / 1024).toFixed(1)}KB`
  } else {
    size = `${sizeBytes}B`
  }

  return {
    id: `custom-${Date.now()}`,
    name,
    description: `Custom uploaded dataset (${text.length.toLocaleString()} characters)`,
    size,
    text,
  }
}

export function splitDataset(
  text: string,
  valFraction = 0.1,
): { train: string; val: string } {
  const splitPoint = Math.floor(text.length * (1 - valFraction))
  return {
    train: text.slice(0, splitPoint),
    val: text.slice(splitPoint),
  }
}
