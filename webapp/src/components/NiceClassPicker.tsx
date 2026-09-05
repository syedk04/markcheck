import { useMemo, useState } from 'preact/hooks'
import { NICE_CLASSES } from '../data/nice-classes'

interface Props {
  selected: number[]
  onChange: (selected: number[]) => void
}

export function NiceClassPicker({ selected, onChange }: Props) {
  const [query, setQuery] = useState('')

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase()
    if (!q) return NICE_CLASSES
    return NICE_CLASSES.filter((c) => c.label.toLowerCase().includes(q) || String(c.number) === q)
  }, [query])

  const toggle = (num: number) => {
    if (selected.includes(num)) {
      onChange(selected.filter((n) => n !== num))
    } else {
      onChange([...selected, num].sort((a, b) => a - b))
    }
  }

  return (
    <div class="panel">
      <h2>Nice classification</h2>
      <p class="field-hint">
        Optional but recommended: select the goods/services classes your mark will be used under. Skipping this
        disables class-overlap scoring in the risk breakdown below.
      </p>
      <input
        type="text"
        placeholder="Search classes (e.g. clothing, 25, software)"
        value={query}
        onInput={(e) => setQuery((e.target as HTMLInputElement).value)}
        class="text-input"
      />
      {selected.length > 0 && (
        <div class="chip-row">
          {selected.map((num) => {
            const cls = NICE_CLASSES.find((c) => c.number === num)!
            return (
              <span class="chip" key={num}>
                {cls.label}
                <button type="button" onClick={() => toggle(num)} aria-label={`Remove ${cls.label}`}>
                  ×
                </button>
              </span>
            )
          })}
        </div>
      )}
      <ul class="class-list">
        {filtered.map((c) => (
          <li key={c.number}>
            <label>
              <input type="checkbox" checked={selected.includes(c.number)} onChange={() => toggle(c.number)} />
              {c.label}
            </label>
          </li>
        ))}
      </ul>
    </div>
  )
}
