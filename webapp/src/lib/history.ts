import type { MatchResult, RiskAssessment } from './types'

export interface HistoryEntry {
  id: string
  timestamp: number
  fileName: string
  selectedClasses: string[]
  results: Array<{ match: MatchResult; risk: RiskAssessment }>
}

const STORAGE_KEY = 'markcheck.history.v1'
const MAX_ENTRIES = 50

export function loadHistory(): HistoryEntry[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw)
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
}

export function saveHistoryEntry(entry: HistoryEntry): void {
  const existing = loadHistory()
  const next = [entry, ...existing].slice(0, MAX_ENTRIES)
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(next))
  } catch {
    // Storage quota exceeded or unavailable (private browsing); history is
    // a convenience feature, so fail silently rather than break search.
  }
}

export function clearHistory(): void {
  localStorage.removeItem(STORAGE_KEY)
}
