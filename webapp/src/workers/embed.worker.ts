/// <reference lib="webworker" />
import { AutoModel, AutoProcessor, CLIPVisionModelWithProjection, env, RawImage } from '@xenova/transformers'
import { pcaProject, quantizeVector } from '../lib/quantize'
import { fourierDescriptor, largestComponentMask, resampleContour, traceBoundary } from '../lib/fourier'
import { prepareImage } from '../lib/imagePrep'
import type { ModelParams } from '../lib/types'

env.allowLocalModels = false
// Threaded WASM requires cross-origin isolation headers this static app does
// not set up; pin to a single thread so the worker loads reliably everywhere.
env.backends.onnx.wasm.numThreads = 1

const DINO_MODEL_ID = 'Xenova/dinov2-small'
const DINO_FALLBACK_MODEL_ID = 'Xenova/dino-vits16'
const CLIP_MODEL_ID = 'Xenova/clip-vit-base-patch32'
const IMAGE_SIZE = 224

interface DinoBundle {
  processor: Awaited<ReturnType<typeof AutoProcessor.from_pretrained>>
  model: Awaited<ReturnType<typeof AutoModel.from_pretrained>>
  modelId: string
  usedFallback: boolean
}

let dinoBundle: DinoBundle | null = null
let clipProcessor: Awaited<ReturnType<typeof AutoProcessor.from_pretrained>> | null = null
let clipModel: Awaited<ReturnType<typeof CLIPVisionModelWithProjection.from_pretrained>> | null = null
let modelParams: ModelParams | null = null
let initPromise: Promise<void> | null = null

type WorkerRequest =
  | { type: 'init' }
  | { type: 'embedQuery'; requestId: string; bitmap: ImageBitmap }
  | { type: 'embedThumbnail'; requestId: string; bitmap: ImageBitmap }

self.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  const msg = event.data
  try {
    if (msg.type === 'init') {
      await ensureInit()
      post({ type: 'ready', usedFallbackDino: dinoBundle!.usedFallback, dinoModelId: dinoBundle!.modelId })
      return
    }
    await ensureInit()
    if (msg.type === 'embedQuery') {
      const result = await embedQuery(msg.bitmap)
      post({ type: 'embedQueryResult', requestId: msg.requestId, ...result }, [
        result.record.dino_color.buffer,
        result.record.dino_grey.buffer,
        result.record.clip.buffer,
        result.record.fourier.buffer,
        result.queryDinoColorRaw.buffer,
      ])
    } else if (msg.type === 'embedThumbnail') {
      const result = await embedThumbnail(msg.bitmap)
      post({ type: 'embedThumbnailResult', requestId: msg.requestId, ...result }, [result.patchTokens.buffer])
    }
  } catch (err) {
    const requestId = 'requestId' in msg ? msg.requestId : undefined
    post({ type: 'error', requestId, message: err instanceof Error ? err.message : String(err) })
  }
}

function post(message: unknown, transfer: Transferable[] = []): void {
  ;(self as unknown as Worker).postMessage(message, transfer)
}

async function ensureInit(): Promise<void> {
  if (!initPromise) {
    initPromise = doInit()
  }
  return initPromise
}

async function doInit(): Promise<void> {
  const paramsRes = await fetch(`${import.meta.env.BASE_URL}data/model_params.json`)
  if (!paramsRes.ok) {
    throw new Error(`Failed to load model_params.json: ${paramsRes.status}`)
  }
  modelParams = (await paramsRes.json()) as ModelParams

  dinoBundle = await loadDino()

  clipProcessor = await AutoProcessor.from_pretrained(CLIP_MODEL_ID)
  clipModel = await CLIPVisionModelWithProjection.from_pretrained(CLIP_MODEL_ID)
}

async function loadDino(): Promise<DinoBundle> {
  try {
    const processor = await AutoProcessor.from_pretrained(DINO_MODEL_ID)
    const model = await AutoModel.from_pretrained(DINO_MODEL_ID)
    return { processor, model, modelId: DINO_MODEL_ID, usedFallback: false }
  } catch (err) {
    console.warn(
      `[markcheck] failed to load ${DINO_MODEL_ID} (${String(err)}); falling back to ${DINO_FALLBACK_MODEL_ID}. ` +
        'This is DINOv1 ViT-S/16, not DINOv2 — flagged so downstream results are not mistaken for the intended backbone.',
    )
    const processor = await AutoProcessor.from_pretrained(DINO_FALLBACK_MODEL_ID)
    const model = await AutoModel.from_pretrained(DINO_FALLBACK_MODEL_ID)
    return { processor, model, modelId: DINO_FALLBACK_MODEL_ID, usedFallback: true }
  }
}

async function runDino(canvas: OffscreenCanvas): Promise<{ cls: Float32Array; patches: Float32Array; numPatches: number; dim: number }> {
  const raw = await canvasToRawImage(canvas)
  const inputs = await dinoBundle!.processor(raw)
  const output = await dinoBundle!.model(inputs)
  const lastHidden = output.last_hidden_state
  const [, seqLen, dim] = lastHidden.dims as [number, number, number]
  const data = lastHidden.data as Float32Array
  const cls = data.slice(0, dim)
  const numPatches = seqLen - 1
  const patches = data.slice(dim, dim * seqLen)
  return { cls: new Float32Array(cls), patches: new Float32Array(patches), numPatches, dim }
}

async function runClip(canvas: OffscreenCanvas): Promise<Float32Array> {
  const raw = await canvasToRawImage(canvas)
  const inputs = await clipProcessor!(raw)
  const output = await clipModel!(inputs)
  const embeds = output.image_embeds.data as Float32Array
  return new Float32Array(embeds)
}

async function canvasToRawImage(canvas: OffscreenCanvas): Promise<RawImage> {
  const ctx = canvas.getContext('2d')!
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)
  const raw = new RawImage(new Uint8ClampedArray(imageData.data), canvas.width, canvas.height, 4)
  return raw.rgb()
}

async function embedQuery(bitmap: ImageBitmap): Promise<{
  record: { dino_color: Int8Array; dino_grey: Int8Array; clip: Int8Array; fourier: Int8Array }
  queryDinoColorRaw: Float32Array
  usedFallbackDino: boolean
  fourierOk: boolean
}> {
  const params = modelParams!
  const prepared = await prepareImage(bitmap, IMAGE_SIZE)

  const [dinoColor, dinoGrey, clipEmbed] = await Promise.all([
    runDino(prepared.colorCanvas),
    runDino(prepared.greyCanvas),
    runClip(prepared.colorCanvas),
  ])

  const dinoColorProjected = pcaProject(dinoColor.cls, params.pca.dino_color!)
  const dinoGreyProjected = pcaProject(dinoGrey.cls, params.pca.dino_grey!)
  const clipProjected = pcaProject(clipEmbed, params.pca.clip!)

  let fourierOk = true
  let fourierRaw: Float64Array
  try {
    const component = largestComponentMask(prepared.mask, prepared.maskWidth, prepared.maskHeight)
    const boundary = traceBoundary(component, prepared.maskWidth, prepared.maskHeight)
    if (boundary.length < 8) throw new Error('contour too small')
    const resampled = resampleContour(boundary, 128)
    fourierRaw = fourierDescriptor(resampled, params.channels.fourier.dim)
  } catch {
    fourierOk = false
    fourierRaw = new Float64Array(params.channels.fourier.dim)
  }

  const record = {
    dino_color: quantizeVector(dinoColorProjected, params.channels.dino_color),
    dino_grey: quantizeVector(dinoGreyProjected, params.channels.dino_grey),
    clip: quantizeVector(clipProjected, params.channels.clip),
    fourier: quantizeVector(fourierRaw, params.channels.fourier),
  }

  return { record, queryDinoColorRaw: dinoColor.cls, usedFallbackDino: dinoBundle!.usedFallback, fourierOk }
}

async function embedThumbnail(
  bitmap: ImageBitmap,
): Promise<{ patchTokens: Float32Array; numPatches: number; dim: number; gridSize: number }> {
  const prepared = await prepareImage(bitmap, IMAGE_SIZE)
  const { patches, numPatches, dim } = await runDino(prepared.colorCanvas)
  const gridSize = Math.round(Math.sqrt(numPatches))
  return { patchTokens: patches, numPatches, dim, gridSize }
}
