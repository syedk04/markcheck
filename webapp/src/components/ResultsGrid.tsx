import type { MatchResult, RiskAssessment } from '../lib/types'
import { RiskBadge } from './RiskBadge'

const dataBase = `${import.meta.env.BASE_URL}data/`

interface Props {
  results: Array<{ match: MatchResult; risk: RiskAssessment }>
  selectedId: string | null
  onSelect: (id: string) => void
}

export function ResultsGrid({ results, selectedId, onSelect }: Props) {
  if (results.length === 0) {
    return <p class="empty-state">No results yet.</p>
  }
  return (
    <div class="results-grid">
      {results.map(({ match, risk }, rank) => (
        <button
          type="button"
          key={match.id}
          class={`result-card ${selectedId === match.id ? 'result-card-selected' : ''}`}
          onClick={() => onSelect(match.id)}
        >
          <span class="result-rank">#{rank + 1}</span>
          <img loading="lazy" src={`${dataBase}${match.thumb}`} alt={match.label} class="result-thumb" />
          <span class="result-label">{match.label}</span>
          <span class="result-id">{match.id}</span>
          <RiskBadge risk={risk} />
          <span class="result-score">visual sim {match.visualSimilarity.toFixed(3)}</span>
        </button>
      ))}
    </div>
  )
}
