import type { HistoryEntry } from '../lib/history'

interface Props {
  entries: HistoryEntry[]
  onClear: () => void
}

export function HistoryPanel({ entries, onClear }: Props) {
  return (
    <div class="panel">
      <div class="panel-header">
        <h2>Search history</h2>
        {entries.length > 0 && (
          <button type="button" onClick={onClear}>
            Clear history
          </button>
        )}
      </div>
      {entries.length === 0 ? (
        <p class="empty-state">No past searches yet. History is stored only in this browser.</p>
      ) : (
        <ul class="history-list">
          {entries.map((entry) => (
            <li key={entry.id}>
              <span class="history-file">{entry.fileName}</span>
              <span class="history-time">{new Date(entry.timestamp).toLocaleString()}</span>
              <span class="history-count">{entry.results.length} matches</span>
              {entry.results[0] && <span class="history-top">top: {entry.results[0].risk.band}</span>}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
