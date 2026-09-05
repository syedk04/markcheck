import { describe, expect, it } from 'vitest'
import { buildChannelScores, classOverlapScore, fuseChannelScores } from '../src/lib/scoring'
import type { ChannelName } from '../src/lib/types'

describe('fuseChannelScores', () => {
  it('computes a weighted average', () => {
    const score = fuseChannelScores([
      { channel: 'dino_color', similarity: 1, weight: 0.5 },
      { channel: 'dino_grey', similarity: 0, weight: 0.2 },
      { channel: 'clip', similarity: 1, weight: 0.2 },
      { channel: 'fourier', similarity: 0, weight: 0.1 },
    ])
    // (1*0.5 + 0*0.2 + 1*0.2 + 0*0.1) / 1.0 = 0.7
    expect(score).toBeCloseTo(0.7)
  })

  it('re-normalizes weights when a channel is missing', () => {
    const score = fuseChannelScores([
      { channel: 'dino_color', similarity: 1, weight: 0.5 },
      { channel: 'dino_grey', similarity: 1, weight: 0.2 },
    ])
    expect(score).toBeCloseTo(1)
  })

  it('returns 0 for an empty score list', () => {
    expect(fuseChannelScores([])).toBe(0)
  })
})

describe('buildChannelScores', () => {
  it('pairs similarities with their configured weight, skipping missing channels', () => {
    const weights: Record<ChannelName, number> = {
      dino_color: 0.5,
      dino_grey: 0.2,
      clip: 0.2,
      fourier: 0.1,
    }
    const scores = buildChannelScores({ dino_color: 0.9, fourier: undefined }, weights)
    expect(scores).toEqual([{ channel: 'dino_color', similarity: 0.9, weight: 0.5 }])
  })
})

describe('classOverlapScore', () => {
  it('returns 0 when no classes are selected', () => {
    expect(classOverlapScore([], 'apparel/footwear')).toBe(0)
  })

  it('matches tokens shared between class labels and the item label', () => {
    const score = classOverlapScore(['Class 25 — Clothing, footwear, headgear'], 'footwear/sneakers')
    expect(score).toBe(1)
  })

  it('returns a fraction when only some selected classes match', () => {
    const score = classOverlapScore(
      ['Class 25 — Clothing, footwear, headgear', 'Class 9 — Computers and software'],
      'footwear/sneakers',
    )
    expect(score).toBeCloseTo(0.5)
  })

  it('returns 0 when nothing matches', () => {
    const score = classOverlapScore(['Class 9 — Computers and software'], 'footwear/sneakers')
    expect(score).toBe(0)
  })
})
