import { useEffect, useRef, useState } from 'preact/hooks'
import type { MatchResult, RiskAssessment } from '../lib/types'
import type { EmbedWorkerClient } from '../lib/workerClient'
import { drawHeatmapOverlay, patchSimilarities } from '../lib/heatmap'
import { RiskBreakdown } from './RiskBadge'

const dataBase = `${import.meta.env.BASE_URL}data/`
const CANVAS_SIZE = 260

interface Props {
  match: MatchResult
  risk: RiskAssessment
  queryPreviewUrl: string
  queryDinoColorRaw: Float32Array | null
  embedClient: EmbedWorkerClient
}

export function ResultDetail({ match, risk, queryPreviewUrl, queryDinoColorRaw, embedClient }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [heatmapState, setHeatmapState] = useState<'idle' | 'loading' | 'ready' | 'error'>('idle')
  const [heatmapError, setHeatmapError] = useState<string | null>(null)

  useEffect(() => {
    setHeatmapState('idle')
    setHeatmapError(null)
  }, [match.id])

  const computeHeatmap = async () => {
    if (!queryDinoColorRaw) {
      setHeatmapState('error')
      setHeatmapError('Query embedding unavailable for this result.')
      return
    }
    setHeatmapState('loading')
    try {
      const imgUrl = `${dataBase}${match.thumb}`
      const img = await loadImage(imgUrl)
      const bitmap = await createImageBitmap(img)
      const { patchTokens, numPatches, dim, gridSize } = await embedClient.embedThumbnail(bitmap)
      const sims = patchSimilarities(patchTokens, numPatches, dim, queryDinoColorRaw)

      const canvas = canvasRef.current!
      canvas.width = CANVAS_SIZE
      canvas.height = CANVAS_SIZE
      const ctx = canvas.getContext('2d')!
      ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE)
      ctx.drawImage(img, 0, 0, CANVAS_SIZE, CANVAS_SIZE)
      drawHeatmapOverlay(ctx, CANVAS_SIZE, CANVAS_SIZE, sims, gridSize)
      setHeatmapState('ready')
    } catch (err) {
      setHeatmapState('error')
      setHeatmapError(err instanceof Error ? err.message : String(err))
    }
  }

  return (
    <div class="panel detail-panel">
      <h2>
        Comparison: your upload vs {match.id} — {match.label}
      </h2>
      <div class="diff-row">
        <figure>
          <img src={queryPreviewUrl} alt="query upload" class="diff-image" />
          <figcaption>Your upload</figcaption>
        </figure>
        <figure>
          <canvas ref={canvasRef} class="diff-image" width={CANVAS_SIZE} height={CANVAS_SIZE} />
          <figcaption>
            Matched mark ({match.id})
            {heatmapState !== 'ready' && (
              <>
                {' '}
                <button type="button" onClick={computeHeatmap} disabled={heatmapState === 'loading'}>
                  {heatmapState === 'loading' ? 'Computing heatmap…' : 'Show match heatmap'}
                </button>
              </>
            )}
          </figcaption>
          {heatmapState === 'error' && <p class="error-text">{heatmapError}</p>}
        </figure>
      </div>

      <h3>Risk reasoning</h3>
      <RiskBreakdown risk={risk} />

      <h3>Per-channel similarity</h3>
      <table class="channel-table">
        <thead>
          <tr>
            <th>Channel</th>
            <th>Weight</th>
            <th>Cosine similarity</th>
          </tr>
        </thead>
        <tbody>
          {match.channelScores.map((c) => (
            <tr key={c.channel}>
              <td>{c.channel}</td>
              <td>{c.weight.toFixed(2)}</td>
              <td>{c.similarity.toFixed(3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p class="field-hint">
        Class overlap is scored against this corpus item's free-text label, not a verified Nice Classification
        assignment — treat it as a rough heuristic, not a legal class match.
      </p>
    </div>
  )
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.onload = () => resolve(img)
    img.onerror = () => reject(new Error(`Failed to load image: ${src}`))
    img.src = src
  })
}
