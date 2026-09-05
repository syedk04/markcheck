import { useEffect, useMemo, useRef, useState } from 'preact/hooks'
import './app.css'
import { Disclaimer } from './components/Disclaimer'
import { UploadPanel } from './components/UploadPanel'
import { NiceClassPicker } from './components/NiceClassPicker'
import { ResultsGrid } from './components/ResultsGrid'
import { ResultDetail } from './components/ResultDetail'
import { BatchTable, type BatchRow } from './components/BatchTable'
import { HistoryPanel } from './components/HistoryPanel'
import { NICE_CLASSES } from './data/nice-classes'
import { EmbedWorkerClient, SearchWorkerClient } from './lib/workerClient'
import { classOverlapScore } from './lib/scoring'
import { computeRisk } from './lib/risk'
import { loadHistory, saveHistoryEntry, clearHistory, type HistoryEntry } from './lib/history'
import { generateReportHtml, generateReportPdf } from './lib/report'
import type { MatchResult, RiskAssessment } from './lib/types'

type JobStatus = 'pending' | 'embedding' | 'searching' | 'done' | 'error'

interface Job {
  id: string
  file: File
  previewUrl: string
  status: JobStatus
  error?: string
  results: Array<{ match: MatchResult; risk: RiskAssessment }>
  queryDinoColorRaw: Float32Array | null
  usedFallbackDino?: boolean
  fourierOk?: boolean
}

const TOP_K = 20

function makeId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`
}

export function App() {
  const [batchMode, setBatchMode] = useState(false)
  const [jobs, setJobs] = useState<Job[]>([])
  const [selectedClasses, setSelectedClasses] = useState<number[]>([])
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null)
  const [selectedMatchId, setSelectedMatchId] = useState<string | null>(null)
  const [runState, setRunState] = useState<'idle' | 'loading-models' | 'running' | 'done'>('idle')
  const [globalError, setGlobalError] = useState<string | null>(null)
  const [history, setHistory] = useState<HistoryEntry[]>([])
  const [modelInfo, setModelInfo] = useState<{ usedFallbackDino: boolean; dinoModelId: string } | null>(null)

  const embedClientRef = useRef<EmbedWorkerClient | null>(null)
  const searchClientRef = useRef<SearchWorkerClient | null>(null)

  useEffect(() => {
    setHistory(loadHistory())
  }, [])

  const selectedClassLabels = useMemo(
    () => selectedClasses.map((n) => NICE_CLASSES.find((c) => c.number === n)!.label),
    [selectedClasses],
  )

  const handleFilesSelected = (files: File[]) => {
    const newJobs: Job[] = files.map((file) => ({
      id: makeId(),
      file,
      previewUrl: URL.createObjectURL(file),
      status: 'pending',
      results: [],
      queryDinoColorRaw: null,
    }))
    setJobs(batchMode ? [...jobs, ...newJobs] : newJobs)
    setSelectedJobId(null)
    setSelectedMatchId(null)
    setRunState('idle')
  }

  const ensureClients = async () => {
    if (!embedClientRef.current) embedClientRef.current = new EmbedWorkerClient()
    if (!searchClientRef.current) searchClientRef.current = new SearchWorkerClient()
    setRunState('loading-models')
    const [info] = await Promise.all([embedClientRef.current.waitUntilReady(), searchClientRef.current.waitUntilReady()])
    setModelInfo(info)
  }

  const runSearches = async () => {
    setGlobalError(null)
    try {
      await ensureClients()
      setRunState('running')
      const embedClient = embedClientRef.current!
      const searchClient = searchClientRef.current!

      for (const job of jobs) {
        if (job.status === 'done') continue
        setJobs((prev) => prev.map((j) => (j.id === job.id ? { ...j, status: 'embedding', error: undefined } : j)))
        try {
          const img = await loadImageFromFile(job.file)
          const bitmap = await createImageBitmap(img)
          const embedResult = await embedClient.embedQuery(bitmap)

          setJobs((prev) => prev.map((j) => (j.id === job.id ? { ...j, status: 'searching' } : j)))
          const matches = await searchClient.search(
            embedResult.record,
            TOP_K,
            embedResult.fourierOk ? undefined : ['fourier'],
          )

          const results = matches.map((match) => {
            const overlap =
              selectedClassLabels.length > 0 ? classOverlapScore(selectedClassLabels, match.label) : null
            const risk = computeRisk(match.visualSimilarity, overlap)
            return { match, risk }
          })

          setJobs((prev) =>
            prev.map((j) =>
              j.id === job.id
                ? {
                    ...j,
                    status: 'done',
                    results,
                    queryDinoColorRaw: embedResult.queryDinoColorRaw,
                    usedFallbackDino: embedResult.usedFallbackDino,
                    fourierOk: embedResult.fourierOk,
                  }
                : j,
            ),
          )

          saveHistoryEntry({
            id: makeId(),
            timestamp: Date.now(),
            fileName: job.file.name,
            selectedClasses: selectedClassLabels,
            results,
          })
        } catch (err) {
          setJobs((prev) =>
            prev.map((j) =>
              j.id === job.id ? { ...j, status: 'error', error: err instanceof Error ? err.message : String(err) } : j,
            ),
          )
        }
      }

      setHistory(loadHistory())
      setRunState('done')
    } catch (err) {
      setGlobalError(err instanceof Error ? err.message : String(err))
      setRunState('idle')
    }
  }

  const selectedJob = jobs.find((j) => j.id === selectedJobId) ?? jobs.find((j) => j.status === 'done') ?? null
  const selectedMatch = selectedJob?.results.find((r) => r.match.id === selectedMatchId) ?? selectedJob?.results[0]

  const batchRows: BatchRow[] = jobs.map((j) => ({
    jobId: j.id,
    fileName: j.file.name,
    status: j.status,
    topMatch: j.results[0] ?? null,
  }))

  const downloadReport = (format: 'pdf' | 'html') => {
    if (!selectedJob) return
    const input = {
      fileName: selectedJob.file.name,
      queryImageDataUrl: null,
      selectedClasses: selectedClassLabels,
      results: selectedJob.results,
      generatedAt: new Date(),
    }
    if (format === 'pdf') {
      const doc = generateReportPdf(input)
      doc.save(`markcheck-report-${selectedJob.file.name}.pdf`)
    } else {
      const html = generateReportHtml(input)
      const blob = new Blob([html], { type: 'text/html' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `markcheck-report-${selectedJob.file.name}.html`
      a.click()
      URL.revokeObjectURL(url)
    }
  }

  return (
    <div class="app-shell">
      <header>
        <h1>MarkCheck</h1>
        <p class="tagline">Client-side trademark visual-similarity screening</p>
      </header>

      <Disclaimer />

      {globalError && <p class="error-text">{globalError}</p>}

      <UploadPanel onFilesSelected={handleFilesSelected} batchMode={batchMode} onBatchModeChange={setBatchMode} />

      {jobs.length > 0 && (
        <div class="panel">
          <h2>Queued uploads</h2>
          <ul class="job-list">
            {jobs.map((j) => (
              <li key={j.id}>
                <img src={j.previewUrl} alt={j.file.name} class="job-thumb" />
                <span>{j.file.name}</span>
                <span class="job-status">{j.status}</span>
                {j.error && <span class="error-text">{j.error}</span>}
              </li>
            ))}
          </ul>
        </div>
      )}

      <NiceClassPicker selected={selectedClasses} onChange={setSelectedClasses} />

      <div class="panel run-panel">
        <button type="button" onClick={runSearches} disabled={jobs.length === 0 || runState === 'running' || runState === 'loading-models'}>
          {runState === 'loading-models'
            ? 'Loading models (first run downloads ~150MB)…'
            : runState === 'running'
              ? 'Searching…'
              : 'Run search'}
        </button>
        {modelInfo?.usedFallbackDino && (
          <p class="warning-text">
            DINOv2 failed to load; fell back to DINOv1 ViT-S/16 ({modelInfo.dinoModelId}). Visual similarity scores
            for this session used the fallback backbone.
          </p>
        )}
      </div>

      <BatchTable rows={batchRows} />

      {selectedJob && selectedJob.results.length > 0 && (
        <div class="panel">
          <div class="panel-header">
            <h2>Results for {selectedJob.file.name}</h2>
            <div>
              <button type="button" onClick={() => downloadReport('pdf')}>
                Download PDF report
              </button>
              <button type="button" onClick={() => downloadReport('html')}>
                Download HTML report
              </button>
            </div>
          </div>
          {selectedJob.fourierOk === false && (
            <p class="warning-text">Contour extraction failed for this upload; the shape (Fourier) channel was skipped.</p>
          )}
          <ResultsGrid
            results={selectedJob.results}
            selectedId={selectedMatch?.match.id ?? null}
            onSelect={setSelectedMatchId}
          />
        </div>
      )}

      {selectedJob && selectedMatch && embedClientRef.current && (
        <ResultDetail
          match={selectedMatch.match}
          risk={selectedMatch.risk}
          queryPreviewUrl={selectedJob.previewUrl}
          queryDinoColorRaw={selectedJob.queryDinoColorRaw}
          embedClient={embedClientRef.current}
        />
      )}

      <HistoryPanel
        entries={history}
        onClear={() => {
          clearHistory()
          setHistory([])
        }}
      />

      <footer>
        <p>All processing runs locally in your browser. No uploads leave your device.</p>
      </footer>
    </div>
  )
}

function loadImageFromFile(file: File): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file)
    const img = new Image()
    img.onload = () => resolve(img)
    img.onerror = () => reject(new Error(`Failed to read image file: ${file.name}`))
    img.src = url
  })
}
