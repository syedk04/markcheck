import type { RiskAssessment, RiskBand } from './types'

// Visual similarity dominates the combined score because it is the only
// channel backed by real corpus data in this build; class overlap is a
// demo-quality heuristic (see scoring.ts) and is weighted down accordingly.
const VISUAL_WEIGHT = 0.7
const CLASS_WEIGHT = 0.3

const HIGH_THRESHOLD = 0.75
const MEDIUM_THRESHOLD = 0.5

export function computeRisk(visualSimilarity: number, classOverlap: number | null): RiskAssessment {
  const combinedScore =
    classOverlap === null ? visualSimilarity : visualSimilarity * VISUAL_WEIGHT + classOverlap * CLASS_WEIGHT

  const band = bandFor(combinedScore)

  return {
    band,
    visualSimilarity,
    classOverlap,
    combinedScore,
  }
}

function bandFor(score: number): RiskBand {
  if (score >= HIGH_THRESHOLD) return 'High'
  if (score >= MEDIUM_THRESHOLD) return 'Medium'
  return 'Low'
}
