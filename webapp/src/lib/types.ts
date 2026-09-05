export type ChannelName = 'dino_color' | 'dino_grey' | 'clip' | 'fourier'

export interface ChannelParams {
  dim: number
  weight: number
  scale: number
  zero_point: number
}

export interface PcaBasis {
  mean: number[]
  components: number[][]
}

export interface ModelParams {
  record_size_bytes: number
  record_order: ChannelName[]
  channels: Record<ChannelName, ChannelParams>
  pca: Partial<Record<ChannelName, PcaBasis>>
}

export interface MetadataItem {
  id: string
  label: string
  thumb: string
}

export interface Metadata {
  items: MetadataItem[]
}

/** Quantized record for one corpus item, keyed by channel name. */
export type QuantizedRecord = Record<ChannelName, Int8Array>

export interface ChannelScore {
  channel: ChannelName
  similarity: number
  weight: number
}

export interface MatchResult {
  id: string
  label: string
  thumb: string
  visualSimilarity: number
  channelScores: ChannelScore[]
}

export type RiskBand = 'Low' | 'Medium' | 'High'

export interface RiskAssessment {
  band: RiskBand
  visualSimilarity: number
  classOverlap: number | null
  combinedScore: number
}
