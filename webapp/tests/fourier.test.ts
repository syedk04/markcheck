import { describe, expect, it } from 'vitest'
import { fourierDescriptor, largestComponentMask, resampleContour, traceBoundary } from '../src/lib/fourier'

function filledCircleMask(size: number, cx: number, cy: number, radius: number): Uint8Array {
  const mask = new Uint8Array(size * size)
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      mask[y * size + x] = Math.hypot(x - cx, y - cy) <= radius ? 255 : 0
    }
  }
  return mask
}

describe('traceBoundary', () => {
  it('returns a closed loop that revisits the start pixel for a filled circle', () => {
    const size = 160
    const mask = filledCircleMask(size, 80, 80, 40)
    const boundary = traceBoundary(mask, size, size)
    expect(boundary.length).toBeGreaterThan(50)
    expect(boundary.length).toBeLessThan(size * size)
    const last = boundary[boundary.length - 1]
    expect(last).toEqual(boundary[0])
  })

  it('terminates instead of looping forever on a filled square', () => {
    const size = 40
    const mask = new Uint8Array(size * size).fill(255)
    const boundary = traceBoundary(mask, size, size)
    expect(boundary.length).toBeGreaterThan(0)
    expect(boundary.length).toBeLessThan(size * size * 8)
  })

  it('returns an empty array for a blank mask', () => {
    const mask = new Uint8Array(100)
    expect(traceBoundary(mask, 10, 10)).toEqual([])
  })
})

describe('largestComponentMask', () => {
  it('keeps only the largest connected blob', () => {
    const size = 20
    const mask = new Uint8Array(size * size)
    // small 2x2 blob
    mask[0] = 255
    mask[1] = 255
    mask[size] = 255
    mask[size + 1] = 255
    // larger 5x5 blob elsewhere
    for (let y = 10; y < 15; y++) {
      for (let x = 10; x < 15; x++) {
        mask[y * size + x] = 255
      }
    }
    const result = largestComponentMask(mask, size, size)
    expect(result[0]).toBe(0)
    expect(result[12 * size + 12]).toBe(255)
  })
})

describe('resampleContour + fourierDescriptor', () => {
  it('produces a scale-invariant descriptor for two circles of different radii', () => {
    const smallMask = filledCircleMask(120, 60, 60, 20)
    const largeMask = filledCircleMask(120, 60, 60, 40)
    const smallBoundary = resampleContour(traceBoundary(smallMask, 120, 120), 128)
    const largeBoundary = resampleContour(traceBoundary(largeMask, 120, 120), 128)
    const smallDescriptor = fourierDescriptor(smallBoundary, 16)
    const largeDescriptor = fourierDescriptor(largeBoundary, 16)

    // Both are near-circular, so the low-frequency shape signature should be
    // close regardless of absolute radius.
    for (let i = 0; i < 16; i++) {
      expect(Math.abs(smallDescriptor[i] - largeDescriptor[i])).toBeLessThan(0.15)
    }
  })

  it('returns zeros when given no contour points', () => {
    const descriptor = fourierDescriptor([], 32)
    expect(Array.from(descriptor)).toEqual(new Array(32).fill(0))
  })
})
