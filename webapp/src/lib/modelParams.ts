import type { ChannelName, ModelParams } from './types'

let cached: ModelParams | null = null

export async function loadModelParams(): Promise<ModelParams> {
  if (cached) return cached
  const res = await fetch(`${import.meta.env.BASE_URL}data/model_params.json`)
  if (!res.ok) {
    throw new Error(`Failed to load model_params.json: ${res.status}`)
  }
  const params = (await res.json()) as ModelParams
  validateModelParams(params)
  cached = params
  return params
}

function validateModelParams(params: ModelParams): void {
  const summedDims = params.record_order.reduce((sum, ch) => sum + params.channels[ch].dim, 0)
  if (summedDims !== params.record_size_bytes) {
    throw new Error(
      `model_params.json is inconsistent: channel dims sum to ${summedDims} bytes but record_size_bytes is ${params.record_size_bytes}`,
    )
  }
}

/** Byte offset of each channel within a fixed-size corpus record. */
export function channelOffsets(params: ModelParams): Record<ChannelName, number> {
  const offsets = {} as Record<ChannelName, number>
  let offset = 0
  for (const channel of params.record_order) {
    offsets[channel] = offset
    offset += params.channels[channel].dim
  }
  return offsets
}
