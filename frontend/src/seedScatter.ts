/** Join predictions to reference CSV rows and build scatter series (committee vs model). */

export type PredictApiRow = {
  RecordID: string
  Team: string
  Predicted_Seed: number
}

export type ScatterPoint = {
  recordId: string
  team: string
  /** True committee / S-curve seed (x, unjittered) */
  committee: number
  /** Model prediction (y, unjittered) */
  predicted: number
  x: number
  y: number
}

export type ScatterDiagnostics = {
  plotted: number
  /** Prediction row has no matching RecordID in reference */
  predMissingRef: number
  /** Reference seed missing or not in 1–68 */
  invalidCommittee: number
  /** Model seed non-finite or out of range */
  invalidPredicted: number
  /** Reference tournament row (valid seed) not in prediction file */
  refNotInPred: number
}

function normalizeId(value: unknown): string {
  if (value == null) return ''
  return String(value).trim()
}

/** Official line: integers 1–68 only (ignores 0 / blanks). */
export function parseCommitteeSeed(raw: string): number | null {
  const t = String(raw ?? '').trim()
  if (!t) return null
  const n = Number.parseInt(t, 10)
  if (!Number.isFinite(n)) return null
  if (n < 1 || n > 68) return null
  return n
}

/** Model output: allow 0 (non-field) through 68; reject junk. */
export function parsePredictedSeed(value: unknown): number | null {
  const n =
    typeof value === 'number' ? value : Number.parseFloat(String(value ?? '').trim())
  if (!Number.isFinite(n)) return null
  if (n < 0 || n > 99) return null
  return Math.round(n)
}

function jitter(id: string, salt: string): number {
  const str = id + salt
  let h = 0
  for (let i = 0; i < str.length; i++) h = (Math.imul(31, h) + str.charCodeAt(i)) | 0
  return ((h % 1000) / 1000 - 0.5) * 0.6
}

export function buildScatterPoints(
  predRows: PredictApiRow[],
  refRows: Record<string, string>[],
  seedCol: string | null,
): { points: ScatterPoint[]; diagnostics: ScatterDiagnostics } {
  const emptyDiag: ScatterDiagnostics = {
    plotted: 0,
    predMissingRef: 0,
    invalidCommittee: 0,
    invalidPredicted: 0,
    refNotInPred: 0,
  }

  if (!seedCol || !predRows.length || !refRows.length) {
    return { points: [], diagnostics: emptyDiag }
  }

  const refById = new Map<string, { committeeRaw: string; team?: string }>()
  for (const r of refRows) {
    const id = normalizeId(r['RecordID'] ?? r['recordid'])
    if (!id) continue
    refById.set(id, {
      committeeRaw: String(r[seedCol] ?? ''),
      team: r['Team'] ? String(r['Team']) : undefined,
    })
  }

  let predMissingRef = 0
  let invalidCommittee = 0
  let invalidPredicted = 0
  const points: ScatterPoint[] = []
  const matchedPredIds = new Set<string>()

  for (const p of predRows) {
    const id = normalizeId(p.RecordID)
    if (!id) continue

    const ref = refById.get(id)
    if (!ref) {
      predMissingRef++
      continue
    }

    const committee = parseCommitteeSeed(ref.committeeRaw)
    if (committee === null) {
      invalidCommittee++
      continue
    }

    const predicted = parsePredictedSeed(p.Predicted_Seed)
    if (predicted === null) {
      invalidPredicted++
      continue
    }

    matchedPredIds.add(id)
    points.push({
      recordId: id,
      team: (p.Team && String(p.Team).trim()) || ref.team || id,
      committee,
      predicted,
      x: committee + jitter(id, 'x'),
      y: predicted + jitter(id, 'y'),
    })
  }

  let refNotInPred = 0
  for (const [id, ref] of refById) {
    if (matchedPredIds.has(id)) continue
    if (parseCommitteeSeed(ref.committeeRaw) !== null) refNotInPred++
  }

  return {
    points,
    diagnostics: {
      plotted: points.length,
      predMissingRef,
      invalidCommittee,
      invalidPredicted,
      refNotInPred,
    },
  }
}
