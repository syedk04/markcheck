import type { MatchResult, RiskAssessment } from '../lib/types'

export interface BatchRow {
  jobId: string
  fileName: string
  status: string
  topMatch: { match: MatchResult; risk: RiskAssessment } | null
}

export function BatchTable({ rows }: { rows: BatchRow[] }) {
  if (rows.length < 2) return null
  return (
    <div class="panel">
      <h2>Batch comparison</h2>
      <table class="batch-table">
        <thead>
          <tr>
            <th>File</th>
            <th>Status</th>
            <th>Top match</th>
            <th>Visual similarity</th>
            <th>Risk</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.jobId}>
              <td>{row.fileName}</td>
              <td>{row.status}</td>
              <td>{row.topMatch ? `${row.topMatch.match.id} — ${row.topMatch.match.label}` : '—'}</td>
              <td>{row.topMatch ? row.topMatch.match.visualSimilarity.toFixed(3) : '—'}</td>
              <td>{row.topMatch ? row.topMatch.risk.band : '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
