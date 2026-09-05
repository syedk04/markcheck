import type { ChannelParams, PcaBasis } from './types'

/**
 * Reconstructs a float value from a stored int8 value. This must match the
 * offline pipeline's quantization formula exactly (value * scale + zero_point)
 * or every downstream similarity score is silently wrong.
 */
export function dequantize(raw: number, params: Pick<ChannelParams, 'scale' | 'zero_point'>): number {
  return raw * params.scale + params.zero_point
}

export function dequantizeVector(raw: Int8Array, params: Pick<ChannelParams, 'scale' | 'zero_point'>): Float64Array {
  const out = new Float64Array(raw.length)
  for (let i = 0; i < raw.length; i++) {
    out[i] = dequantize(raw[i], params)
  }
  return out
}

/** Inverse of dequantize, clamped to the int8 range, rounded to nearest integer. */
export function quantize(value: number, params: Pick<ChannelParams, 'scale' | 'zero_point'>): number {
  const raw = Math.round((value - params.zero_point) / params.scale)
  if (raw > 127) return 127
  if (raw < -128) return -128
  return raw
}

export function quantizeVector(values: ArrayLike<number>, params: Pick<ChannelParams, 'scale' | 'zero_point'>): Int8Array {
  const out = new Int8Array(values.length)
  for (let i = 0; i < values.length; i++) {
    out[i] = quantize(values[i], params)
  }
  return out
}

/**
 * Projects a raw embedding into the reduced PCA space: mean-subtract then
 * multiply by the component matrix. `components` is stored as
 * (reducedDim x originalDim), so each output coordinate is the dot product
 * of the mean-centered input with one row of components.
 */
export function pcaProject(raw: ArrayLike<number>, basis: PcaBasis): Float64Array {
  const { mean, components } = basis
  if (raw.length !== mean.length) {
    throw new Error(`pcaProject: input dim ${raw.length} does not match basis mean dim ${mean.length}`)
  }
  const centered = new Float64Array(raw.length)
  for (let i = 0; i < raw.length; i++) {
    centered[i] = raw[i] - mean[i]
  }
  const reducedDim = components.length
  const out = new Float64Array(reducedDim)
  for (let j = 0; j < reducedDim; j++) {
    const row = components[j]
    let sum = 0
    for (let i = 0; i < row.length; i++) {
      sum += row[i] * centered[i]
    }
    out[j] = sum
  }
  return out
}

export function cosineSimilarity(a: ArrayLike<number>, b: ArrayLike<number>): number {
  if (a.length !== b.length) {
    throw new Error(`cosineSimilarity: length mismatch ${a.length} vs ${b.length}`)
  }
  let dot = 0
  let normA = 0
  let normB = 0
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i]
    normA += a[i] * a[i]
    normB += b[i] * b[i]
  }
  if (normA === 0 || normB === 0) return 0
  return dot / (Math.sqrt(normA) * Math.sqrt(normB))
}
