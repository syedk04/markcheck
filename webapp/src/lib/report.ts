import { jsPDF } from 'jspdf'
import type { MatchResult, RiskAssessment } from './types'

export interface ReportInput {
  fileName: string
  queryImageDataUrl: string | null
  selectedClasses: string[]
  results: Array<{ match: MatchResult; risk: RiskAssessment }>
  generatedAt: Date
}

const DISCLAIMER =
  'This report is a preliminary, automated visual similarity screening. It is not legal advice ' +
  'and is not a substitute for a professional clearance search or the opinion of a trademark attorney.'

export function generateReportPdf(input: ReportInput): jsPDF {
  const doc = new jsPDF({ unit: 'pt', format: 'a4' })
  const marginX = 40
  let y = 48

  doc.setFontSize(18)
  doc.text('MarkCheck Screening Report', marginX, y)
  y += 22

  doc.setFontSize(10)
  doc.setTextColor(120, 20, 20)
  const disclaimerLines = doc.splitTextToSize(DISCLAIMER, 515)
  doc.text(disclaimerLines, marginX, y)
  y += disclaimerLines.length * 12 + 12
  doc.setTextColor(0, 0, 0)

  doc.setFontSize(11)
  doc.text(`Query file: ${input.fileName}`, marginX, y)
  y += 16
  doc.text(`Generated: ${input.generatedAt.toLocaleString()}`, marginX, y)
  y += 16
  doc.text(
    `Selected Nice classes: ${input.selectedClasses.length > 0 ? input.selectedClasses.join(', ') : 'none selected'}`,
    marginX,
    y,
    { maxWidth: 515 },
  )
  y += 24

  if (input.queryImageDataUrl) {
    try {
      doc.addImage(input.queryImageDataUrl, 'PNG', marginX, y, 80, 80)
    } catch {
      // Non-fatal: some browsers may produce a data URL format jsPDF can't
      // parse from a canvas export; the report still lists match data.
    }
    y += 96
  }

  doc.setFontSize(13)
  doc.text('Results', marginX, y)
  y += 18
  doc.setFontSize(10)

  for (const { match, risk } of input.results) {
    if (y > 760) {
      doc.addPage()
      y = 48
    }
    doc.setFont('helvetica', 'bold')
    doc.text(`${match.id} — ${match.label}`, marginX, y)
    doc.setFont('helvetica', 'normal')
    y += 14
    doc.text(
      `Risk: ${risk.band}  |  Visual similarity: ${risk.visualSimilarity.toFixed(3)}  |  ` +
        `Class overlap (heuristic): ${risk.classOverlap === null ? 'n/a' : risk.classOverlap.toFixed(3)}  |  ` +
        `Combined: ${risk.combinedScore.toFixed(3)}`,
      marginX,
      y,
    )
    y += 14
    const channelLine = match.channelScores.map((c) => `${c.channel}=${c.similarity.toFixed(3)}`).join('  ')
    doc.text(`Channel similarities: ${channelLine}`, marginX, y)
    y += 20
  }

  return doc
}

export function generateReportHtml(input: ReportInput): string {
  const rows = input.results
    .map(
      ({ match, risk }) => `
      <tr>
        <td>${match.id}</td>
        <td>${escapeHtml(match.label)}</td>
        <td>${risk.band}</td>
        <td>${risk.visualSimilarity.toFixed(3)}</td>
        <td>${risk.classOverlap === null ? 'n/a' : risk.classOverlap.toFixed(3)}</td>
        <td>${risk.combinedScore.toFixed(3)}</td>
        <td>${match.channelScores.map((c) => `${c.channel}=${c.similarity.toFixed(3)}`).join(', ')}</td>
      </tr>`,
    )
    .join('')

  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>MarkCheck Report — ${escapeHtml(input.fileName)}</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 32px; color: #1a1a1a; }
  .disclaimer { background: #fdecec; border: 1px solid #d33; padding: 12px 16px; border-radius: 6px; margin-bottom: 24px; }
  table { border-collapse: collapse; width: 100%; }
  th, td { border: 1px solid #ccc; padding: 6px 10px; text-align: left; font-size: 13px; }
  th { background: #f2f2f2; }
</style>
</head>
<body>
  <h1>MarkCheck Screening Report</h1>
  <div class="disclaimer"><strong>Not legal advice.</strong> ${DISCLAIMER}</div>
  <p><strong>Query file:</strong> ${escapeHtml(input.fileName)}</p>
  <p><strong>Generated:</strong> ${input.generatedAt.toLocaleString()}</p>
  <p><strong>Selected Nice classes:</strong> ${
    input.selectedClasses.length > 0 ? input.selectedClasses.map(escapeHtml).join(', ') : 'none selected'
  }</p>
  ${input.queryImageDataUrl ? `<img src="${input.queryImageDataUrl}" alt="query" style="max-width:160px;" />` : ''}
  <h2>Results</h2>
  <table>
    <thead>
      <tr><th>ID</th><th>Label</th><th>Risk</th><th>Visual similarity</th><th>Class overlap</th><th>Combined</th><th>Channel similarities</th></tr>
    </thead>
    <tbody>${rows}</tbody>
  </table>
</body>
</html>`
}

function escapeHtml(text: string): string {
  return text.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]!)
}
