import {
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  LabelList,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { ScatterDiagnostics, ScatterPoint } from './seedScatter'
import './SeedScatterChart.css'

type Props = {
  points: ScatterPoint[]
  diagnostics: ScatterDiagnostics
  seedColName: string
}

export function SeedScatterChart({ points, diagnostics, seedColName }: Props) {
  if (!points.length) {
    return (
      <div className="scatter-panel scatter-panel--empty">
        <h2>Committee vs predicted seed</h2>
        <p className="scatter-placeholder">
          No overlapping teams with a valid committee seed (1–68) and model output. Upload
          predictions and ensure the reference file uses matching <code>RecordID</code> values.
        </p>
        {(diagnostics.predMissingRef > 0 ||
          diagnostics.invalidCommittee > 0 ||
          diagnostics.invalidPredicted > 0 ||
          diagnostics.refNotInPred > 0) && (
          <ul className="scatter-diag">
            {diagnostics.predMissingRef > 0 && (
              <li>
                Predictions with no reference row: <strong>{diagnostics.predMissingRef}</strong>
              </li>
            )}
            {diagnostics.invalidCommittee > 0 && (
              <li>
                Reference rows with missing/invalid seed (need 1–68):{' '}
                <strong>{diagnostics.invalidCommittee}</strong>
              </li>
            )}
            {diagnostics.invalidPredicted > 0 && (
              <li>
                Non-numeric or out-of-range predicted seeds:{' '}
                <strong>{diagnostics.invalidPredicted}</strong>
              </li>
            )}
            {diagnostics.refNotInPred > 0 && (
              <li>
                Committee field teams missing from prediction file:{' '}
                <strong>{diagnostics.refNotInPred}</strong>
              </li>
            )}
          </ul>
        )}
      </div>
    )
  }

  return (
    <div className="scatter-panel">
      <h2>Committee vs predicted seed</h2>
      <p className="scatter-sub">
        Each point is one team: horizontal axis = reference <code>{seedColName}</code> (committee),
        vertical = model <code>Predicted_Seed</code>. The dashed line is perfect agreement (y = x).
        Slight position jitter separates exact ties.
      </p>

      <div className="scatter-chart-wrap">
        <ResponsiveContainer width="100%" height={520}>
          <ScatterChart margin={{ top: 24, right: 42, bottom: 20, left: 12 }}>
            <CartesianGrid strokeDasharray="4 4" stroke="#cbd5e1" />
            <XAxis
              type="number"
              dataKey="x"
              name="Committee"
              domain={[0, 70]}
              ticks={[0, 17, 34, 51, 68]}
              stroke="#94a3b8"
              tick={{ fill: '#1d4ed8', fontSize: 12, fontWeight: 600 }}
              label={{
                value: 'Committee seed (reference)',
                position: 'bottom',
                offset: 0,
                fill: '#1d4ed8',
              }}
            />
            <YAxis
              type="number"
              dataKey="y"
              name="Predicted"
              domain={[0, 70]}
              ticks={[0, 17, 34, 51, 68]}
              stroke="#94a3b8"
              tick={{ fill: '#1d4ed8', fontSize: 12, fontWeight: 600 }}
              label={{
                value: 'Model predicted seed',
                angle: -90,
                position: 'insideLeft',
                fill: '#1d4ed8',
              }}
            />
            <Tooltip
              cursor={{ strokeDasharray: '4 4', stroke: '#2563eb' }}
              content={({ active, payload }) => {
                if (!active || !payload?.length) return null
                const d = payload[0].payload as ScatterPoint
                const delta = d.predicted - d.committee
                return (
                  <div className="scatter-tooltip">
                    <div>
                      Committee: <strong>{d.committee}</strong>
                    </div>
                    <div>
                      Predicted: <strong>{d.predicted}</strong>
                    </div>
                    <div>
                      Δ (pred − committee):{' '}
                      <strong>{delta > 0 ? `+${delta}` : String(delta)}</strong>
                    </div>
                  </div>
                )
              }}
            />
            <ReferenceLine
              segment={[
                { x: 1, y: 1 },
                { x: 68, y: 68 },
              ]}
              stroke="#60a5fa"
              strokeWidth={1.5}
              strokeDasharray="6 4"
              ifOverflow="visible"
            />
            <Scatter name="Teams" data={points} fill="#2563eb" fillOpacity={0.9}>
              <LabelList
                dataKey="team"
                position="top"
                fill="#1d4ed8"
                fontSize={10}
                formatter={(value) => {
                  const label = String(value ?? '')
                  return label.length > 14 ? `${label.slice(0, 14)}…` : label
                }}
              />
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
