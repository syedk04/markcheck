import { cosineSimilarity } from './quantize'

/**
 * Per-patch cosine similarity between each DINO patch token of the matched
 * image and the query's pooled (CLS) embedding. High values mark the region
 * of the matched image that most resembles the overall query mark.
 */
export function patchSimilarities(
  patchTokens: Float32Array,
  numPatches: number,
  dim: number,
  queryEmbedding: Float32Array,
): Float64Array {
  const out = new Float64Array(numPatches)
  for (let p = 0; p < numPatches; p++) {
    const patch = patchTokens.subarray(p * dim, (p + 1) * dim)
    out[p] = cosineSimilarity(patch, queryEmbedding)
  }
  return out
}

/** Rescales similarities to 0..1 so the weakest patch in-frame is not blank. */
export function normalizeToUnitRange(values: Float64Array): Float64Array {
  let min = Infinity
  let max = -Infinity
  for (const v of values) {
    if (v < min) min = v
    if (v > max) max = v
  }
  const range = max - min
  const out = new Float64Array(values.length)
  for (let i = 0; i < values.length; i++) {
    out[i] = range > 1e-9 ? (values[i] - min) / range : 0.5
  }
  return out
}

/** Simple blue -> yellow -> red heat colormap. */
export function heatColor(t: number): [number, number, number] {
  const clamped = Math.min(1, Math.max(0, t))
  if (clamped < 0.5) {
    const k = clamped / 0.5
    return [Math.round(30 + k * 200), Math.round(60 + k * 180), Math.round(200 - k * 150)]
  }
  const k = (clamped - 0.5) / 0.5
  return [Math.round(230 + k * 25), Math.round(240 - k * 200), Math.round(50 - k * 50)]
}

export function drawHeatmapOverlay(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  values: Float64Array,
  gridSize: number,
  alpha = 0.55,
): void {
  const normalized = normalizeToUnitRange(values)
  const cellW = width / gridSize
  const cellH = height / gridSize
  ctx.save()
  ctx.globalAlpha = alpha
  for (let row = 0; row < gridSize; row++) {
    for (let col = 0; col < gridSize; col++) {
      const idx = row * gridSize + col
      if (idx >= normalized.length) continue
      const [r, g, b] = heatColor(normalized[idx])
      ctx.fillStyle = `rgb(${r},${g},${b})`
      ctx.fillRect(col * cellW, row * cellH, cellW, cellH)
    }
  }
  ctx.restore()
}
