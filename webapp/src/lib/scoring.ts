import type { ChannelName, ChannelScore } from './types'

/**
 * Combines per-channel cosine similarities into a single visual similarity
 * score using the channel weights from model_params.json. Weights are
 * re-normalized over the channels actually present so a missing channel
 * (e.g. the in-browser contour tracer failing to produce a Fourier
 * descriptor for a particular upload) degrades gracefully instead of
 * silently deflating the overall score.
 */
export function fuseChannelScores(scores: ChannelScore[]): number {
  if (scores.length === 0) return 0
  const totalWeight = scores.reduce((sum, s) => sum + s.weight, 0)
  if (totalWeight === 0) return 0
  const weighted = scores.reduce((sum, s) => sum + s.similarity * s.weight, 0)
  return weighted / totalWeight
}

export function buildChannelScores(
  similarities: Partial<Record<ChannelName, number>>,
  weights: Record<ChannelName, number>,
): ChannelScore[] {
  const out: ChannelScore[] = []
  for (const channel of Object.keys(similarities) as ChannelName[]) {
    const similarity = similarities[channel]
    if (similarity === undefined) continue
    out.push({ channel, similarity, weight: weights[channel] })
  }
  return out
}

/**
 * Heuristic class-overlap score against whatever free-text label the
 * corpus item carries. This is explicitly demo-quality: the placeholder
 * (and likely early real) corpus does not carry true Nice Classification
 * labels, so this only measures token overlap between the user-selected
 * class descriptions and the item's label string.
 */
export function classOverlapScore(selectedClassLabels: string[], itemLabel: string): number {
  if (selectedClassLabels.length === 0) return 0
  const itemTokens = tokenize(itemLabel)
  if (itemTokens.size === 0) return 0
  let matched = 0
  for (const classLabel of selectedClassLabels) {
    const classTokens = tokenize(classLabel)
    let hit = false
    for (const token of classTokens) {
      if (itemTokens.has(token)) {
        hit = true
        break
      }
    }
    if (hit) matched++
  }
  return matched / selectedClassLabels.length
}

function tokenize(text: string): Set<string> {
  return new Set(
    text
      .toLowerCase()
      .split(/[^a-z0-9]+/)
      .filter((t) => t.length > 2),
  )
}
