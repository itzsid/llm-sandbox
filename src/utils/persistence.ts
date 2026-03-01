const STORAGE_KEY = 'llm-sandbox-state'

export interface SavedState {
  configText: string
  selectedDatasetId: string | null
  configSubTab: string
  timestamp: number
}

export function saveState(state: SavedState): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state))
  } catch {
    // quota exceeded or unavailable
  }
}

export function loadState(): SavedState | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return null
    return JSON.parse(raw) as SavedState
  } catch {
    return null
  }
}

export function clearState(): void {
  try {
    localStorage.removeItem(STORAGE_KEY)
  } catch {
    // ignore
  }
}
