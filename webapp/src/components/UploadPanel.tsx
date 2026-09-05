import { useRef, useState } from 'preact/hooks'

interface Props {
  onFilesSelected: (files: File[]) => void
  batchMode: boolean
  onBatchModeChange: (batch: boolean) => void
}

export function UploadPanel({ onFilesSelected, batchMode, onBatchModeChange }: Props) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragOver, setDragOver] = useState(false)

  const handleFiles = (fileList: FileList | null) => {
    if (!fileList || fileList.length === 0) return
    const files = Array.from(fileList).filter((f) => f.type.startsWith('image/'))
    if (files.length > 0) onFilesSelected(batchMode ? files : [files[0]])
  }

  return (
    <div class="panel">
      <div class="panel-header">
        <h2>Upload</h2>
        <label class="batch-toggle">
          <input
            type="checkbox"
            checked={batchMode}
            onChange={(e) => onBatchModeChange((e.target as HTMLInputElement).checked)}
          />
          Batch mode (multiple variants)
        </label>
      </div>
      <div
        class={`dropzone ${dragOver ? 'dropzone-active' : ''}`}
        onDragOver={(e) => {
          e.preventDefault()
          setDragOver(true)
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDragOver(false)
          handleFiles(e.dataTransfer?.files ?? null)
        }}
        onClick={() => inputRef.current?.click()}
      >
        <p>Drag and drop a logo image here, or click to browse.</p>
        <p class="dropzone-hint">{batchMode ? 'Multiple files supported.' : 'One file at a time.'}</p>
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          multiple={batchMode}
          style={{ display: 'none' }}
          onChange={(e) => handleFiles((e.target as HTMLInputElement).files)}
        />
      </div>
    </div>
  )
}
