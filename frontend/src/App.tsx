import { useCallback, useEffect, useMemo, useState } from 'react'
import Papa from 'papaparse'
import './App.css'

const API_BASE = import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8000'

/** Served from frontend/public/reference/ for instant demo compare */
const BUNDLED_REFERENCE_CSV = '/reference/tournament_seeds_2025-26.csv'

type PredictApiRow = {
  RecordID: string
  Season: string
  Team: string
  Predicted_Seed: number
  Blended_Model_Score: number | null
}

type PredictResponse = {
  scenario: string
  rows: PredictApiRow[]
  n_rows: number
}

function useDragHighlight() {
  const [active, setActive] = useState(false)
  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setActive(true)
  }, [])
  const onDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setActive(false)
  }, [])
  return { active, onDragOver, onDragLeave, setActive }
}

function pickSeedColumn(headers: string[]): string | null {
  const lower = headers.map((h) => h.trim().toLowerCase())
  const idx = lower.findIndex(
    (h) =>
      h === 'overall seed' ||
      h === 'official seed' ||
      h === 'true seed' ||
      h === 'seed',
  )
  if (idx >= 0) return headers[idx].trim()
  return null
}

function App() {
  const [predRows, setPredRows] = useState<PredictApiRow[]>([])
  const [predFileName, setPredFileName] = useState<string | null>(null)
  const [predLoading, setPredLoading] = useState(false)
  const [predError, setPredError] = useState<string | null>(null)
  const [scenario, setScenario] = useState('future_season')
  const [trainFile, setTrainFile] = useState<File | null>(null)

  const [refRows, setRefRows] = useState<Record<string, string>[]>([])
  const [refFileName, setRefFileName] = useState<string | null>(null)
  const [refError, setRefError] = useState<string | null>(null)
  const [refPreloaded, setRefPreloaded] = useState(false)

  const leftDrop = useDragHighlight()
  const rightDrop = useDragHighlight()

  const refColumns = useMemo(() => {
    if (!refRows.length) return []
    return Object.keys(refRows[0] ?? {})
  }, [refRows])

  const matchStats = useMemo(() => {
    if (!predRows.length || !refRows.length || !refColumns.length) return null
    const seedCol = pickSeedColumn(refColumns)
    if (!seedCol) return null
    const refMap = new Map<string, string>()
    for (const r of refRows) {
      const id = r['RecordID'] ?? r['recordid']
      if (id != null && id !== '') refMap.set(String(id).trim(), String(r[seedCol] ?? ''))
    }
    let compared = 0
    let matches = 0
    for (const p of predRows) {
      const official = refMap.get(String(p.RecordID).trim())
      if (official === undefined) continue
      compared += 1
      if (String(p.Predicted_Seed) === official.trim()) matches += 1
    }
    return { compared, matches, seedCol }
  }, [predRows, refRows, refColumns])

  const applyReferenceText = (text: string, label: string) => {
    setRefError(null)
    setRefFileName(label)
    Papa.parse<Record<string, string>>(text, {
      header: true,
      skipEmptyLines: true,
      complete: (results) => {
        if (results.errors.length) {
          setRefError(results.errors.map((x) => x.message).join('; '))
        }
        const rows = (results.data as Record<string, string>[]).filter(
          (r) => Object.keys(r).some((k) => r[k]),
        )
        setRefRows(rows)
        const headers = rows[0] ? Object.keys(rows[0]) : []
        if (rows.length && !pickSeedColumn(headers)) {
          setRefError(
            'No seed column found (expected Overall Seed, Seed, etc.). Showing raw columns.',
          )
        }
      },
      error: (err: Error) => {
        setRefRows([])
        setRefError(err.message)
      },
    })
  }

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const res = await fetch(BUNDLED_REFERENCE_CSV)
        if (!res.ok || cancelled) return
        const text = await res.text()
        if (cancelled) return
        applyReferenceText(text, 'tournament_seeds_2025-26.csv (bundled)')
        setRefPreloaded(true)
      } catch {
        /* dev server or missing public file — user can still drop a CSV */
      }
    })()
    return () => {
      cancelled = true
    }
  }, [])

  const runPredict = async (file: File) => {
    if (scenario === 'historical_test' && !trainFile) {
      setPredError('For historical_test, choose a training CSV (reference for constrained seeds) first.')
      return
    }
    setPredError(null)
    setPredLoading(true)
    setPredFileName(file.name)
    const fd = new FormData()
    fd.append('file', file)
    fd.append('scenario', scenario)
    if (scenario === 'historical_test' && trainFile) {
      fd.append('train_file', trainFile)
    }
    try {
      const res = await fetch(`${API_BASE}/api/predict`, {
        method: 'POST',
        body: fd,
      })
      if (!res.ok) {
        const text = await res.text()
        throw new Error(text || `HTTP ${res.status}`)
      }
      const data: PredictResponse = await res.json()
      setPredRows(data.rows ?? [])
    } catch (e) {
      setPredRows([])
      setPredError(e instanceof Error ? e.message : String(e))
    } finally {
      setPredLoading(false)
    }
  }

  const onDropPredict = async (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    leftDrop.setActive(false)
    const f = e.dataTransfer.files?.[0]
    if (f) void runPredict(f)
  }

  const onFileInputPredict = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]
    if (f) void runPredict(f)
  }

  const parseReferenceFile = (file: File) => {
    setRefError(null)
    setRefFileName(file.name)
    Papa.parse<Record<string, string>>(file, {
      header: true,
      skipEmptyLines: true,
      complete: (results) => {
        if (results.errors.length) {
          setRefError(results.errors.map((x) => x.message).join('; '))
        }
        const rows = (results.data as Record<string, string>[]).filter(
          (r) => Object.keys(r).some((k) => r[k]),
        )
        setRefRows(rows)
        const headers = rows[0] ? Object.keys(rows[0]) : []
        if (rows.length && !pickSeedColumn(headers)) {
          setRefError(
            'No seed column found (expected Overall Seed, Seed, etc.). Showing raw columns.',
          )
        }
      },
      error: (err: Error) => {
        setRefRows([])
        setRefError(err.message)
      },
    })
  }

  const onDropReference = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    rightDrop.setActive(false)
    const f = e.dataTransfer.files?.[0]
    if (f) parseReferenceFile(f)
  }

  const onFileInputReference = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]
    if (f) parseReferenceFile(f)
  }

  /** Tournament field only: hide non-tournament rows (predicted seed 0). */
  const displayPred = useMemo(() => {
    const field = predRows.filter((r) => (r.Predicted_Seed ?? 0) > 0)
    return field.sort(
      (a, b) => (a.Predicted_Seed || 0) - (b.Predicted_Seed || 0),
    )
  }, [predRows])

  const displayRef = useMemo(() => {
    const seedCol = pickSeedColumn(refColumns)
    if (!seedCol) return refRows
    return [...refRows].sort((a, b) => {
      const sa = parseInt(String(a[seedCol] ?? '999'), 10) || 999
      const sb = parseInt(String(b[seedCol] ?? '999'), 10) || 999
      return sa - sb
    })
  }, [refRows, refColumns])

  return (
    <div className="page">
      <header className="header">
        <h1>NCAA seed prediction — presentation</h1>
        <p className="sub">
          Left: model output on unseen season data via API (
          <code>future_season</code>). Right: S-curve-style reference (bundled{' '}
          <code>tournament_seeds_2025-26.csv</code> loads automatically; override by dropping your own CSV).
        </p>
        <div className="toolbar">
          <label className="scenario-label">
            Scenario{' '}
            <select
              value={scenario}
              onChange={(e) => setScenario(e.target.value)}
              disabled={predLoading}
            >
              <option value="future_season">future_season (2025–26 style)</option>
              <option value="historical_test">historical_test (Kaggle-style + train CSV)</option>
            </select>
          </label>
          {scenario === 'historical_test' && (
            <label className="train-file-label">
              Train CSV{' '}
              <input
                type="file"
                accept=".csv"
                disabled={predLoading}
                onChange={(e) => setTrainFile(e.target.files?.[0] ?? null)}
              />
            </label>
          )}
          <span className="api-hint">
            API: <code>{API_BASE}</code>
          </span>
        </div>
      </header>

      {matchStats && (
        <div className="stats-banner">
          Compared on <code>RecordID</code> vs <code>{matchStats.seedCol}</code>:{' '}
          <strong>{matchStats.matches}</strong> matches / <strong>{matchStats.compared}</strong>{' '}
          overlapping teams
        </div>
      )}

      <div className="grid">
        <section
          className={`panel ${leftDrop.active ? 'drop-active' : ''}`}
          onDragOver={leftDrop.onDragOver}
          onDragLeave={leftDrop.onDragLeave}
          onDrop={onDropPredict}
        >
          <h2>Model predictions</h2>
          <p className="hint">Drag & drop your input CSV, or choose a file.</p>
          <input type="file" accept=".csv" onChange={onFileInputPredict} />
          {predLoading && <p className="loading">Running model…</p>}
          {predError && <p className="error">{predError}</p>}
          {predFileName && !predLoading && (
            <p className="meta">
              File: <strong>{predFileName}</strong> — {displayPred.length} tournament teams
              {predRows.length !== displayPred.length && (
                <span> ({predRows.length} rows incl. seed 0)</span>
              )}
            </p>
          )}
          <div className="table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>RecordID</th>
                  <th>Team</th>
                  <th>Seed</th>
                </tr>
              </thead>
              <tbody>
                {displayPred.map((r) => (
                  <tr key={r.RecordID}>
                    <td className="mono">{r.RecordID}</td>
                    <td>{r.Team}</td>
                    <td className="num">{r.Predicted_Seed}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {!predRows.length && !predLoading && (
              <p className="placeholder">No predictions yet.</p>
            )}
            {!!predRows.length && !displayPred.length && !predLoading && (
              <p className="placeholder">No tournament teams (all seeds 0).</p>
            )}
          </div>
        </section>

        <section
          className={`panel ${rightDrop.active ? 'drop-active' : ''}`}
          onDragOver={rightDrop.onDragOver}
          onDragLeave={rightDrop.onDragLeave}
          onDrop={onDropReference}
        >
          <h2>Reference (internet / official)</h2>
          <p className="hint">
            {refPreloaded
              ? 'Bundled 2025–26 reference is loaded below. Drop another CSV to replace.'
              : 'Drag & drop a CSV with RecordID + Seed (or Overall Seed).'}
          </p>
          <input type="file" accept=".csv" onChange={onFileInputReference} />
          {refError && <p className="error soft">{refError}</p>}
          {refFileName && (
            <p className="meta">
              File: <strong>{refFileName}</strong> — {refRows.length} rows
            </p>
          )}
          <div className="table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  {refColumns.slice(0, 6).map((c) => (
                    <th key={c}>{c}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {displayRef.map((r, i) => (
                  <tr key={i}>
                    {refColumns.slice(0, 6).map((c) => (
                      <td key={c} className={c.toLowerCase().includes('seed') ? 'num' : ''}>
                        {r[c]}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
            {!refRows.length && (
              <p className="placeholder">Load a reference file to compare.</p>
            )}
          </div>
        </section>
      </div>
    </div>
  )
}

export default App
