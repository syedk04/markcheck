/// <reference lib="webworker" />
import { loadModelParams } from '../lib/modelParams'
import { loadCorpus, channelSlice, recordCount, type Corpus } from '../lib/corpus'
import { cosineSimilarity, dequantizeVector } from '../lib/quantize'
import { buildChannelScores, fuseChannelScores } from '../lib/scoring'
import type { ChannelName, MatchResult, ModelParams } from '../lib/types'

let corpus: Corpus | null = null
let params: ModelParams | null = null

type WorkerRequest =
  | { type: 'init' }
  | {
      type: 'search'
      requestId: string
      queryRecord: Record<ChannelName, Int8Array>
      topK: number
      excludeChannels?: ChannelName[]
    }

self.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  const msg = event.data
  try {
    if (msg.type === 'init') {
      params = await loadModelParams()
      corpus = await loadCorpus(params)
      ;(self as unknown as Worker).postMessage({ type: 'ready', itemCount: recordCount(corpus) })
      return
    }
    if (msg.type === 'search') {
      if (!corpus || !params) throw new Error('search worker used before init')
      const matches = search(corpus, params, msg.queryRecord, msg.topK, msg.excludeChannels)
      ;(self as unknown as Worker).postMessage({ type: 'searchResult', requestId: msg.requestId, matches })
    }
  } catch (err) {
    const requestId = 'requestId' in msg ? msg.requestId : undefined
    ;(self as unknown as Worker).postMessage({
      type: 'error',
      requestId,
      message: err instanceof Error ? err.message : String(err),
    })
  }
}

function search(
  corpus: Corpus,
  params: ModelParams,
  queryRecord: Record<ChannelName, Int8Array>,
  topK: number,
  excludeChannels?: ChannelName[],
): MatchResult[] {
  const excluded = new Set(excludeChannels ?? [])
  const channels = params.record_order.filter((c) => !excluded.has(c))
  const queryFloat: Record<ChannelName, Float64Array> = {} as Record<ChannelName, Float64Array>
  for (const channel of channels) {
    queryFloat[channel] = dequantizeVector(queryRecord[channel], params.channels[channel])
  }

  const weights = {} as Record<ChannelName, number>
  for (const channel of channels) {
    weights[channel] = params.channels[channel].weight
  }

  const n = recordCount(corpus)
  const scored: MatchResult[] = new Array(n)

  for (let i = 0; i < n; i++) {
    const similarities: Partial<Record<ChannelName, number>> = {}
    for (const channel of channels) {
      const corpusVec = dequantizeVector(channelSlice(corpus, i, channel), params.channels[channel])
      similarities[channel] = cosineSimilarity(queryFloat[channel], corpusVec)
    }
    const channelScores = buildChannelScores(similarities, weights)
    const visualSimilarity = fuseChannelScores(channelScores)
    const item = corpus.metadata.items[i]
    scored[i] = {
      id: item.id,
      label: item.label,
      thumb: item.thumb,
      visualSimilarity,
      channelScores,
    }
  }

  scored.sort((a, b) => b.visualSimilarity - a.visualSimilarity)
  return scored.slice(0, topK)
}
