import { describe, expect, it } from 'vitest'
import { computeRisk } from '../src/lib/risk'

describe('computeRisk', () => {
  it('bands High when combined score is very high', () => {
    const r = computeRisk(0.95, 1)
    expect(r.band).toBe('High')
    expect(r.combinedScore).toBeCloseTo(0.95 * 0.7 + 1 * 0.3)
  })

  it('bands Medium at a mid-range combined score', () => {
    const r = computeRisk(0.6, 0.5)
    expect(r.combinedScore).toBeCloseTo(0.6 * 0.7 + 0.5 * 0.3)
    expect(r.band).toBe('Medium')
  })

  it('bands Low when combined score is small', () => {
    const r = computeRisk(0.1, 0)
    expect(r.band).toBe('Low')
  })

  it('falls back to visual similarity alone when class overlap is null', () => {
    const r = computeRisk(0.8, null)
    expect(r.combinedScore).toBe(0.8)
    expect(r.band).toBe('High')
  })

  it('always exposes both sub-scores alongside the band', () => {
    const r = computeRisk(0.42, 0.1)
    expect(r).toHaveProperty('visualSimilarity', 0.42)
    expect(r).toHaveProperty('classOverlap', 0.1)
    expect(r).toHaveProperty('combinedScore')
    expect(r).toHaveProperty('band')
  })
})
