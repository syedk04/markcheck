// Client-side Fourier descriptor pipeline. The offline corpus pipeline uses a
// dedicated raster-to-vector tracer to extract contours; running that same
// tool in a browser worker is not practical (native/WASI-threads packaging
// risk), so this reimplements a simplified contour trace plus a plain DFT.
// It is a documented asymmetry against the offline pipeline, not a bug.

export interface Point {
  x: number
  y: number
}

/** 4-connected flood fill to find the largest foreground component's pixels. */
export function largestComponentMask(mask: Uint8Array, width: number, height: number): Uint8Array {
  const visited = new Uint8Array(width * height)
  let bestPixels: number[] | null = null

  for (let start = 0; start < mask.length; start++) {
    if (mask[start] === 0 || visited[start]) continue
    const stack = [start]
    visited[start] = 1
    const component: number[] = []
    while (stack.length > 0) {
      const idx = stack.pop()!
      component.push(idx)
      const x = idx % width
      const y = Math.floor(idx / width)
      const neighbors = [
        [x - 1, y],
        [x + 1, y],
        [x, y - 1],
        [x, y + 1],
      ]
      for (const [nx, ny] of neighbors) {
        if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue
        const nIdx = ny * width + nx
        if (mask[nIdx] !== 0 && !visited[nIdx]) {
          visited[nIdx] = 1
          stack.push(nIdx)
        }
      }
    }
    if (!bestPixels || component.length > bestPixels.length) {
      bestPixels = component
    }
  }

  const out = new Uint8Array(width * height)
  if (bestPixels) {
    for (const idx of bestPixels) out[idx] = 255
  }
  return out
}

// 8-connected neighbor offsets in clockwise order starting at West. Index
// arithmetic mod 8 lets the tracer resume scanning right after the pixel it
// arrived from without a special case for each of the 8 directions.
const NEIGHBOR_OFFSETS: Point[] = [
  { x: -1, y: 0 }, // W
  { x: -1, y: -1 }, // NW
  { x: 0, y: -1 }, // N
  { x: 1, y: -1 }, // NE
  { x: 1, y: 0 }, // E
  { x: 1, y: 1 }, // SE
  { x: 0, y: 1 }, // S
  { x: -1, y: 1 }, // SW
]

/**
 * Moore-neighbor boundary tracing: walks the outer boundary of a single
 * connected foreground blob starting from its topmost-leftmost pixel, using
 * Jacob's stopping criterion (stop on returning to the start pixel via the
 * same entering step that began the trace).
 */
export function traceBoundary(mask: Uint8Array, width: number, height: number): Point[] {
  const isFg = (x: number, y: number): boolean => {
    if (x < 0 || y < 0 || x >= width || y >= height) return false
    return mask[y * width + x] !== 0
  }

  let startX = -1
  let startY = -1
  outer: for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (isFg(x, y)) {
        startX = x
        startY = y
        break outer
      }
    }
  }
  if (startX === -1) return []

  // A lone foreground pixel with no neighbors has no meaningful boundary.
  let hasNeighbor = false
  for (const off of NEIGHBOR_OFFSETS) {
    if (isFg(startX + off.x, startY + off.y)) {
      hasNeighbor = true
      break
    }
  }
  if (!hasNeighbor) return [{ x: startX, y: startY }]

  const points: Point[] = [{ x: startX, y: startY }]
  let cx = startX
  let cy = startY
  // Backtrack starts one step West of the start pixel (guaranteed background
  // since scanning found the start pixel left-to-right, top-to-bottom).
  let backDir = 0
  const maxSteps = width * height * 8

  for (let steps = 0; steps < maxSteps; steps++) {
    let found = false
    for (let i = 1; i <= 8; i++) {
      const dir = (backDir + i) % 8
      const nx = cx + NEIGHBOR_OFFSETS[dir].x
      const ny = cy + NEIGHBOR_OFFSETS[dir].y
      if (isFg(nx, ny)) {
        cx = nx
        cy = ny
        backDir = (dir + 4) % 8
        found = true
        break
      }
    }
    if (!found) break
    points.push({ x: cx, y: cy })
    if (cx === startX && cy === startY) break
  }

  return points
}

/** Resamples a closed polyline to N points spaced evenly by arc length. */
export function resampleContour(points: Point[], n: number): Point[] {
  if (points.length === 0) return Array.from({ length: n }, () => ({ x: 0, y: 0 }))
  if (points.length === 1) return Array.from({ length: n }, () => points[0])

  const closed = [...points, points[0]]
  const cumLength: number[] = [0]
  for (let i = 1; i < closed.length; i++) {
    const dx = closed[i].x - closed[i - 1].x
    const dy = closed[i].y - closed[i - 1].y
    cumLength.push(cumLength[i - 1] + Math.hypot(dx, dy))
  }
  const total = cumLength[cumLength.length - 1]
  if (total === 0) return Array.from({ length: n }, () => points[0])

  const out: Point[] = []
  for (let k = 0; k < n; k++) {
    const target = (k / n) * total
    let seg = 0
    while (seg < cumLength.length - 1 && cumLength[seg + 1] < target) seg++
    const segStart = cumLength[seg]
    const segEnd = cumLength[seg + 1] ?? segStart
    const t = segEnd > segStart ? (target - segStart) / (segEnd - segStart) : 0
    const a = closed[seg]
    const b = closed[seg + 1] ?? closed[seg]
    out.push({ x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t })
  }
  return out
}

/**
 * Computes a rotation/translation/scale-invariant Fourier descriptor from a
 * closed contour: DC term (translation) is dropped, magnitude spectrum
 * (rotation and starting-point invariant) is taken for the first `count`
 * harmonics, and normalized by the first harmonic's magnitude (scale
 * invariance).
 */
export function fourierDescriptor(points: Point[], count: number): Float64Array {
  const n = points.length
  const out = new Float64Array(count)
  if (n === 0) return out

  const magnitudes: number[] = []
  for (let k = 1; k <= count; k++) {
    let re = 0
    let im = 0
    for (let t = 0; t < n; t++) {
      const angle = (-2 * Math.PI * k * t) / n
      const cos = Math.cos(angle)
      const sin = Math.sin(angle)
      re += points[t].x * cos - points[t].y * sin
      im += points[t].x * sin + points[t].y * cos
    }
    magnitudes.push(Math.hypot(re, im) / n)
  }

  const norm = magnitudes[0] || 1
  for (let i = 0; i < count; i++) {
    out[i] = magnitudes[i] / norm
  }
  return out
}
