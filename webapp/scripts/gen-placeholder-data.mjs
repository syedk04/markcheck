// Generates synthetic dev/test data matching the frozen data contract so the
// webapp can be built and smoke-tested before the real corpus pipeline
// produces webapp/public/data. Not used by the production build itself.
import { mkdir, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import sharp from 'sharp'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const dataDir = path.join(__dirname, '..', 'public', 'data')
const thumbDir = path.join(dataDir, 'thumbnails')

const ITEM_COUNT = 200
const CLUSTER_COUNT = 20
const RECORD_ORDER = ['dino_color', 'dino_grey', 'clip', 'fourier']
const CHANNELS = {
  dino_color: { dim: 128, weight: 0.5, scale: 0.01, zero_point: 0, rawDim: 384 },
  dino_grey: { dim: 128, weight: 0.2, scale: 0.01, zero_point: 0, rawDim: 384 },
  clip: { dim: 64, weight: 0.2, scale: 0.01, zero_point: 0, rawDim: 512 },
  fourier: { dim: 32, weight: 0.1, scale: 0.01, zero_point: 0 },
}
const RECORD_SIZE_BYTES = RECORD_ORDER.reduce((sum, ch) => sum + CHANNELS[ch].dim, 0)

const LABEL_CATEGORIES = [
  'apparel/footwear',
  'apparel/outerwear',
  'food/beverage',
  'food/confectionery',
  'technology/software',
  'technology/hardware',
  'automotive/parts',
  'finance/services',
  'entertainment/media',
  'household/furniture',
]

function seededRandom(seed) {
  let state = seed >>> 0
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0
    return state / 0xffffffff
  }
}

function randomUnitVector(rand, dim) {
  const v = new Float64Array(dim)
  for (let i = 0; i < dim; i++) v[i] = rand() * 2 - 1
  let norm = 0
  for (let i = 0; i < dim; i++) norm += v[i] * v[i]
  norm = Math.sqrt(norm) || 1
  for (let i = 0; i < dim; i++) v[i] /= norm
  return v
}

function quantize(value, scale, zeroPoint) {
  const raw = Math.round((value - zeroPoint) / scale)
  return Math.max(-128, Math.min(127, raw))
}

async function main() {
  await mkdir(thumbDir, { recursive: true })
  const rand = seededRandom(42)

  // --- model_params.json ---------------------------------------------
  const pca = {}
  for (const ch of ['dino_color', 'dino_grey', 'clip']) {
    const { dim, rawDim } = CHANNELS[ch]
    const mean = Array.from({ length: rawDim }, () => (rand() - 0.5) * 0.1)
    const components = Array.from({ length: dim }, () => Array.from({ length: rawDim }, () => (rand() - 0.5) * 0.05))
    pca[ch] = { mean, components }
  }
  const modelParams = {
    record_size_bytes: RECORD_SIZE_BYTES,
    record_order: RECORD_ORDER,
    channels: Object.fromEntries(
      RECORD_ORDER.map((ch) => [ch, { dim: CHANNELS[ch].dim, weight: CHANNELS[ch].weight, scale: CHANNELS[ch].scale, zero_point: CHANNELS[ch].zero_point }]),
    ),
    pca,
  }
  await writeFile(path.join(dataDir, 'model_params.json'), JSON.stringify(modelParams))

  // --- corpus: clustered synthetic embeddings -------------------------
  const clusterPrototypes = Array.from({ length: CLUSTER_COUNT }, () =>
    Object.fromEntries(RECORD_ORDER.map((ch) => [ch, randomUnitVector(rand, CHANNELS[ch].dim)])),
  )

  const items = []
  const recordBuffer = Buffer.alloc(ITEM_COUNT * RECORD_SIZE_BYTES)

  for (let i = 0; i < ITEM_COUNT; i++) {
    const clusterIdx = i % CLUSTER_COUNT
    const proto = clusterPrototypes[clusterIdx]
    const id = String(i + 1).padStart(6, '0')
    const category = LABEL_CATEGORIES[clusterIdx % LABEL_CATEGORIES.length]
    items.push({ id, label: `${category}/variant-${clusterIdx}`, thumb: `thumbnails/${id}.jpg` })

    let offset = i * RECORD_SIZE_BYTES
    for (const ch of RECORD_ORDER) {
      const { dim, scale, zero_point } = CHANNELS[ch]
      for (let d = 0; d < dim; d++) {
        const noise = (rand() - 0.5) * 0.3
        const value = proto[ch][d] + noise
        recordBuffer.writeInt8(quantize(value, scale, zero_point), offset + d)
      }
      offset += dim
    }
  }

  await writeFile(path.join(dataDir, 'metadata.json'), JSON.stringify({ items }))
  await writeFile(path.join(dataDir, 'index.bin'), recordBuffer)

  // --- thumbnails: solid color squares, one per cluster hue -----------
  const thumbTasks = items.map((item, i) => {
    const clusterIdx = i % CLUSTER_COUNT
    const hue = Math.round((clusterIdx / CLUSTER_COUNT) * 360)
    const { r, g, b } = hslToRgb(hue, 0.55, 0.6)
    return sharp({ create: { width: 64, height: 64, channels: 3, background: { r, g, b } } })
      .jpeg({ quality: 80 })
      .toFile(path.join(thumbDir, `${item.id}.jpg`))
  })
  await Promise.all(thumbTasks)

  console.log(`Generated ${ITEM_COUNT} placeholder records (${RECORD_SIZE_BYTES} bytes each) and thumbnails in ${dataDir}`)
}

function hslToRgb(h, s, l) {
  const c = (1 - Math.abs(2 * l - 1)) * s
  const x = c * (1 - Math.abs(((h / 60) % 2) - 1))
  const m = l - c / 2
  let r = 0
  let g = 0
  let b = 0
  if (h < 60) [r, g, b] = [c, x, 0]
  else if (h < 120) [r, g, b] = [x, c, 0]
  else if (h < 180) [r, g, b] = [0, c, x]
  else if (h < 240) [r, g, b] = [0, x, c]
  else if (h < 300) [r, g, b] = [x, 0, c]
  else [r, g, b] = [c, 0, x]
  return { r: Math.round((r + m) * 255), g: Math.round((g + m) * 255), b: Math.round((b + m) * 255) }
}

main().catch((err) => {
  console.error(err)
  process.exit(1)
})
