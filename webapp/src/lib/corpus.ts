import type { ChannelName, Metadata, ModelParams } from './types'
import { channelOffsets } from './modelParams'

export interface Corpus {
  metadata: Metadata
  recordBytes: Uint8Array
  params: ModelParams
  offsets: Record<ChannelName, number>
}

let cached: Corpus | null = null

export async function loadCorpus(params: ModelParams): Promise<Corpus> {
  if (cached) return cached
  const base = import.meta.env.BASE_URL
  const [metaRes, binRes] = await Promise.all([fetch(`${base}data/metadata.json`), fetch(`${base}data/index.bin`)])
  if (!metaRes.ok) throw new Error(`Failed to load metadata.json: ${metaRes.status}`)
  if (!binRes.ok) throw new Error(`Failed to load index.bin: ${binRes.status}`)
  const metadata = (await metaRes.json()) as Metadata
  const buffer = await binRes.arrayBuffer()
  const recordBytes = new Uint8Array(buffer)

  const expectedItems = recordBytes.length / params.record_size_bytes
  if (!Number.isInteger(expectedItems)) {
    throw new Error(
      `index.bin size (${recordBytes.length} bytes) is not a multiple of record_size_bytes (${params.record_size_bytes})`,
    )
  }
  if (expectedItems !== metadata.items.length) {
    throw new Error(
      `metadata.json has ${metadata.items.length} items but index.bin implies ${expectedItems} records`,
    )
  }

  cached = { metadata, recordBytes, params, offsets: channelOffsets(params) }
  return cached
}

export function recordCount(corpus: Corpus): number {
  return corpus.metadata.items.length
}

export function channelSlice(corpus: Corpus, itemIndex: number, channel: ChannelName): Int8Array {
  const recordStart = itemIndex * corpus.params.record_size_bytes
  const channelStart = recordStart + corpus.offsets[channel]
  const dim = corpus.params.channels[channel].dim
  // Int8Array view over the underlying buffer's bytes, reinterpreted as signed.
  return new Int8Array(corpus.recordBytes.buffer, corpus.recordBytes.byteOffset + channelStart, dim)
}
