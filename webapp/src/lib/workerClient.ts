import type { ChannelName, MatchResult } from './types'

export interface EmbedQueryResult {
  record: Record<ChannelName, Int8Array>
  queryDinoColorRaw: Float32Array
  usedFallbackDino: boolean
  fourierOk: boolean
}

export interface EmbedThumbnailResult {
  patchTokens: Float32Array
  numPatches: number
  dim: number
  gridSize: number
}

type Pending = { resolve: (value: unknown) => void; reject: (err: Error) => void }

function makeRequestId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`
}

export class EmbedWorkerClient {
  private worker: Worker
  private pending = new Map<string, Pending>()
  private readyPromise: Promise<{ usedFallbackDino: boolean; dinoModelId: string }>
  private resolveReady!: (v: { usedFallbackDino: boolean; dinoModelId: string }) => void
  private rejectReady!: (err: Error) => void

  constructor() {
    this.worker = new Worker(new URL('../workers/embed.worker.ts', import.meta.url), { type: 'module' })
    this.readyPromise = new Promise((resolve, reject) => {
      this.resolveReady = resolve
      this.rejectReady = reject
    })
    this.worker.onmessage = (event) => this.handleMessage(event.data)
    this.worker.onerror = (event) => this.rejectReady(new Error(event.message ?? 'embed worker crashed'))
    this.worker.postMessage({ type: 'init' })
  }

  private handleMessage(data: any): void {
    if (data.type === 'ready') {
      this.resolveReady({ usedFallbackDino: data.usedFallbackDino, dinoModelId: data.dinoModelId })
      return
    }
    if (data.type === 'error' && data.requestId === undefined) {
      // Failure during model init (no in-flight request to attach it to).
      this.rejectReady(new Error(data.message))
      return
    }
    const pending = this.pending.get(data.requestId)
    if (!pending) return
    this.pending.delete(data.requestId)
    if (data.type === 'error') {
      pending.reject(new Error(data.message))
    } else {
      pending.resolve(data)
    }
  }

  waitUntilReady(): Promise<{ usedFallbackDino: boolean; dinoModelId: string }> {
    return this.readyPromise
  }

  embedQuery(bitmap: ImageBitmap): Promise<EmbedQueryResult> {
    const requestId = makeRequestId()
    return new Promise((resolve, reject) => {
      this.pending.set(requestId, { resolve: resolve as (v: unknown) => void, reject })
      this.worker.postMessage({ type: 'embedQuery', requestId, bitmap }, [bitmap])
    })
  }

  embedThumbnail(bitmap: ImageBitmap): Promise<EmbedThumbnailResult> {
    const requestId = makeRequestId()
    return new Promise((resolve, reject) => {
      this.pending.set(requestId, { resolve: resolve as (v: unknown) => void, reject })
      this.worker.postMessage({ type: 'embedThumbnail', requestId, bitmap }, [bitmap])
    })
  }

  terminate(): void {
    this.worker.terminate()
  }
}

export class SearchWorkerClient {
  private worker: Worker
  private pending = new Map<string, Pending>()
  private readyPromise: Promise<{ itemCount: number }>
  private resolveReady!: (v: { itemCount: number }) => void
  private rejectReady!: (err: Error) => void

  constructor() {
    this.worker = new Worker(new URL('../workers/search.worker.ts', import.meta.url), { type: 'module' })
    this.readyPromise = new Promise((resolve, reject) => {
      this.resolveReady = resolve
      this.rejectReady = reject
    })
    this.worker.onmessage = (event) => this.handleMessage(event.data)
    this.worker.onerror = (event) => this.rejectReady(new Error(event.message ?? 'search worker crashed'))
    this.worker.postMessage({ type: 'init' })
  }

  private handleMessage(data: any): void {
    if (data.type === 'ready') {
      this.resolveReady({ itemCount: data.itemCount })
      return
    }
    if (data.type === 'error' && data.requestId === undefined) {
      this.rejectReady(new Error(data.message))
      return
    }
    const pending = this.pending.get(data.requestId)
    if (!pending) return
    this.pending.delete(data.requestId)
    if (data.type === 'error') {
      pending.reject(new Error(data.message))
    } else {
      pending.resolve(data)
    }
  }

  waitUntilReady(): Promise<{ itemCount: number }> {
    return this.readyPromise
  }

  search(queryRecord: Record<ChannelName, Int8Array>, topK: number): Promise<MatchResult[]> {
    const requestId = makeRequestId()
    return new Promise((resolve, reject) => {
      this.pending.set(requestId, { resolve: (v) => resolve((v as any).matches), reject })
      this.worker.postMessage({ type: 'search', requestId, queryRecord, topK })
    })
  }

  terminate(): void {
    this.worker.terminate()
  }
}
