import type { RiskAssessment } from '../lib/types'

export function RiskBadge({ risk }: { risk: RiskAssessment }) {
  return (
    <span class={`risk-badge risk-${risk.band.toLowerCase()}`}>
      {risk.band} risk
    </span>
  )
}

export function RiskBreakdown({ risk }: { risk: RiskAssessment }) {
  return (
    <dl class="risk-breakdown">
      <dt>Visual similarity</dt>
      <dd>{risk.visualSimilarity.toFixed(3)}</dd>
      <dt>Class overlap (heuristic)</dt>
      <dd>{risk.classOverlap === null ? 'not scored — no classes selected' : risk.classOverlap.toFixed(3)}</dd>
      <dt>Combined score</dt>
      <dd>{risk.combinedScore.toFixed(3)}</dd>
    </dl>
  )
}
