import { describe, expect, it } from 'vitest'
import { cosineSimilarity, dequantize, dequantizeVector, pcaProject, quantize, quantizeVector } from '../src/lib/quantize'

describe('dequantize', () => {
  it('applies value * scale + zero_point', () => {
    expect(dequantize(50, { scale: 0.01, zero_point: 0 })).toBeCloseTo(0.5)
    expect(dequantize(-100, { scale: 0.02, zero_point: 0.1 })).toBeCloseTo(-1.9)
  })

  it('round-trips through quantize within scale precision', () => {
    const params = { scale: 0.01, zero_point: 0 }
    for (const v of [0, 0.5, -0.5, 1.27, -1.28]) {
      const q = quantize(v, params)
      const back = dequantize(q, params)
      expect(back).toBeCloseTo(v, 2)
    }
  })

  it('clamps quantize output to int8 range', () => {
    const params = { scale: 0.01, zero_point: 0 }
    expect(quantize(100, params)).toBe(127)
    expect(quantize(-100, params)).toBe(-128)
  })

  it('dequantizes whole vectors element-wise', () => {
    const raw = new Int8Array([0, 10, -10, 127, -128])
    const out = dequantizeVector(raw, { scale: 0.01, zero_point: 0 })
    expect(Array.from(out)).toEqual([0, 0.1, -0.1, 1.27, -1.28])
  })

  it('quantizeVector is the inverse of dequantizeVector for exact multiples of scale', () => {
    const params = { scale: 0.01, zero_point: 0 }
    const values = [0, 0.1, -0.1, 1.27, -1.28]
    const q = quantizeVector(values, params)
    expect(Array.from(q)).toEqual([0, 10, -10, 127, -128])
  })
})

describe('pcaProject', () => {
  it('mean-subtracts then multiplies by the component matrix', () => {
    // 3-dim raw space, reduced to 2 dims
    const mean = [1, 2, 3]
    const components = [
      [1, 0, 0],
      [0, 1, 0],
    ]
    const raw = [4, 6, 3]
    const projected = pcaProject(raw, { mean, components })
    // centered = [3, 4, 0]
    // row0 . centered = 3, row1 . centered = 4
    expect(Array.from(projected)).toEqual([3, 4])
  })

  it('throws on dimension mismatch', () => {
    expect(() => pcaProject([1, 2], { mean: [1, 2, 3], components: [[1, 1, 1]] })).toThrow()
  })
})

describe('cosineSimilarity', () => {
  it('returns 1 for identical vectors', () => {
    expect(cosineSimilarity([1, 2, 3], [1, 2, 3])).toBeCloseTo(1)
  })

  it('returns 0 for orthogonal vectors', () => {
    expect(cosineSimilarity([1, 0], [0, 1])).toBeCloseTo(0)
  })

  it('returns -1 for opposite vectors', () => {
    expect(cosineSimilarity([1, 2], [-1, -2])).toBeCloseTo(-1)
  })

  it('returns 0 when either vector is all zero', () => {
    expect(cosineSimilarity([0, 0], [1, 2])).toBe(0)
  })
})
