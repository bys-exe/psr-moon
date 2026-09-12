import { Suspense, lazy, useRef, useState } from 'react'
import './App.css'

const Moon3D = lazy(() => import('./Moon3D.jsx'))

const ACCEPT = '.jpg,.jpeg,.png,.tif,.tiff,.webp'
const ALLOWED_EXTS = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp']
const MAX_BYTES = 50 * 1024 * 1024
const BACKEND_OFFLINE =
  'Unable to connect to the backend. Please check whether FastAPI is running.'
const FORMAT_ERROR =
  'Unsupported format. Please upload JPG, JPEG, PNG, TIFF, or WEBP (50 MB max).'

function VisHistogram({ before, after }) {
  const W = 280
  const H = 110
  const PAD = 4
  const STEP = 2
  const n = Math.ceil(256 / STEP)
  const sums = []
  let peak = 1
  for (let b = 0; b < n; b++) {
    let bv = 0
    let av = 0
    for (let k = 0; k < STEP && b * STEP + k < 256; k++) {
      bv += before[b * STEP + k] || 0
      av += after[b * STEP + k] || 0
    }
    sums.push([bv, av])
    peak = Math.max(peak, bv, av)
  }
  // Shared log scale: the crushed-black spike stays visible without
  // flattening everything else.
  const scale = (H - PAD * 2) / Math.log1p(peak)
  const bw = (W - PAD * 2) / n
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="vis-hist"
      role="img"
      aria-label="Brightness histogram before and after enhancement"
    >
      {sums.map(([bv, av], b) => {
        const x = PAD + b * bw
        const w = Math.max(0.5, bw - 0.5)
        return (
          <g key={b}>
            <rect
              x={x}
              y={H - PAD - Math.log1p(bv) * scale}
              width={w}
              height={Math.log1p(bv) * scale}
              fill="#5b6b82"
              opacity="0.85"
            />
            <rect
              x={x}
              y={H - PAD - Math.log1p(av) * scale}
              width={w}
              height={Math.log1p(av) * scale}
              fill="#7dd3fc"
              opacity="0.7"
            />
          </g>
        )
      })}
    </svg>
  )
}

const TI_LEGENDS = {
  shadow: {
    title: 'Shadow colors',
    items: [
      ['#00aa00', 'Illuminated — direct light, directly observed'],
      ['#ffd700', 'Transition — partial light'],
      ['#1e1e1e', 'Shadow — observable darkness, NOT proven PSR'],
      ['#ff8c00', 'Orange boundary — candidate PSR edge'],
    ],
  },
  hazard: {
    title: 'Hazard colors',
    items: [
      ['#00b400', 'Low risk — gentle, well-seen terrain'],
      ['#ffd700', 'Moderate risk — rough or near rims'],
      ['#ff0000', 'High risk — steep, rough or broken ground'],
      ['#969696', 'Unknown — shadowed and featureless, no data'],
    ],
  },
  route: {
    title: 'Markers & route',
    items: [
      ['#2850ff', 'Rover start point (you click it)'],
      ['#e63c3c', 'Destination target (you click it)'],
      ['#ffff78', 'Planned route line'],
    ],
  },
}

function LegendStrip({ group }) {
  const g = TI_LEGENDS[group]
  if (!g) return null
  return (
    <div className="legend-strip" aria-label={g.title}>
      <span className="legend-strip-title">{g.title}</span>
      {g.items.map(([color, text]) => (
        <span key={text} className="legend-strip-item">
          <i style={{ background: color }} />
          {text}
        </span>
      ))}
    </div>
  )
}

const TI_ALGOS = {
  shadows: {
    title: 'Algorithms used — PSR (Permanently Shadowed Region) analysis',
    items: [
      ['Gaussian blur (5×5)', 'Averages each pixel with its neighbors using bell-curve weights. This melts sensor grain so single noisy pixels can’t pass as shadow blobs later.'],
      ["Otsu's method", 'Tries every possible brightness cutoff and keeps the one that splits the histogram into the most compact dark and bright classes (minimum within-class variance). The threshold adapts to each image — nothing hand-tuned.'],
      ['Dual-threshold fusion', 'Pixels below 0.80× the Otsu value = shadow core; 0.80–1.25× = transition ring (penumbra); the rest = illuminated. The ring matters because real shadow edges fade gradually, not in one step.'],
      ['Morphological opening + closing', 'Opening (shrink-then-grow with a 5×5 elliptical kernel) deletes specks smaller than the kernel; closing (grow-then-shrink, 9×9) fills pinholes inside shadow blobs. Elliptical kernels avoid favoring any direction.'],
      ['Connected-component labeling', 'Flood-fills adjacent shadow pixels into blobs (8-connectivity). Each blob is measured — area, centroid, bounding box — and blobs under 25 px (or 0.02% of the image) are discarded as noise. Survivors become the candidate regions in the table.'],
      ['Compactness (4πA/P²)', 'Area vs contour perimeter: 1.0 = perfect circle, low values = ragged or streaky shapes. Round, compact darkness is more consistent with a crater-floor cold trap.'],
    ],
    foot: 'Dark = candidate shadow region only. Permanence is UNKNOWN without illumination time-series data.',
  },
  hazards: {
    title: 'Algorithms used — Hazards',
    items: [
      ['Sobel gradient magnitude', 'Two 3×3 kernels measure how fast brightness changes horizontally and vertically; the combined strength is the slope proxy. Steep sun-facing slopes light up, flat mare stays dark.'],
      ['Local roughness', 'Standard deviation of brightness inside a 15×15 sliding window. Smooth plains score near zero; rubble, ridges and highlands score high — regardless of overall brightness.'],
      ['Euclidean (L2) distance transform', 'Measures every pixel’s straight-line distance to the nearest crater-rim pixel, then flips it into a proximity score that fades to zero at 10% of the image size. Cells hugging rims score high because rims mean steep, blocky ground.'],
      ['Percentile normalization', 'Slope, roughness and proximity live in different units, so each is rescaled to 0–1 using its 1st and 99th percentiles. Outlier pixels can’t hijack the scale.'],
      ['Weighted sum + thresholds', 'Risk = w₁·slope + w₂·roughness + w₃·rim-proximity + w₄·shadow-uncertainty (weights shown as chips, auto-normalized to sum 1). Below 0.33 = LOW, above 0.66 = HIGH. Shadowed *and* featureless cells become UNKNOWN — there is simply no data to judge them.'],
    ],
    foot: 'Weights are shown as chips above; the score is a transparent linear combination, not a black box.',
  },
  route: {
    title: 'Algorithms used — Route',
    items: [
      ['Cost raster', 'Every pixel gets a traversal price: 1 + hazard-penalty + unknown-penalty + shadow-penalty. Cheap pixels are flat, well-seen ground; pricey ones are steep, rough, dark, or unknown.'],
      ['Presets × caution', 'Shortest uses tiny penalties so the path stays near-geometric; Safest multiplies them up to 8× so the path detours around risk; Balanced sits between. The caution dial (0–3) scales all penalties, and cost is clamped to 1–50.'],
      ['A* (A-star) / Dijkstra’s algorithm', 'Both expand the cheapest-known frontier cell first using a binary-heap priority queue. A* additionally guesses remaining cost as straight-line distance × cheapest possible price (an admissible guess, so the result stays optimal) and is much faster; Dijkstra uses no guess and explores uniformly. Grid is 8-connected (diagonals cost √2), capped at 1.5M expansions.'],
      ['Chaikin’s corner-cutting', 'Each grid step is kinked at 45°/90°, so the display curve is smoothed by repeatedly cutting corners at the ¼ and ¾ points (2 passes). Display only — distance and hazard stats always come from the true unsmoothed grid path.'],
    ],
    foot: 'Routes are ESTIMATED from image-derived costs — no DEM (Digital Elevation Model), so never true terrain distance.',
  },
}

function AlgoDetails({ tab }) {
  const g = TI_ALGOS[tab]
  if (!g) return null
  return (
    <details className="manual-more">
      <summary>{g.title}</summary>
      <ul>
        {g.items.map(([name, desc]) => (
          <li key={name}>
            <strong>{name}</strong> — {desc}
          </li>
        ))}
      </ul>
      {g.foot && <p className="note">{g.foot}</p>}
    </details>
  )
}

function RoutePath({ result, dims }) {
  // Vector route line in analysis pixels, drawn OVER the chosen underlay
  // (hazard / shadow) so the viewer shows WHY this path is optimal.
  // Scales with the image because the SVG shares the overlay's viewBox.
  const line =
    (result.display_path && result.display_path.length >= 2
      ? result.display_path
      : result.path) || []
  if (!dims || line.length < 2) return null
  const pts = line.map(([x, y]) => `${x},${y}`).join(' ')
  const sw = Math.max(2, dims.w / 300)
  return (
    <svg
      viewBox={`0 0 ${dims.w} ${dims.h}`}
      className="ti-overlay"
      preserveAspectRatio="none"
      aria-hidden="true"
    >
      <polyline
        points={pts}
        fill="none"
        stroke="#ffff78"
        strokeWidth={sw}
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    </svg>
  )
}

function extOf(name) {
  const n = (name || '').toLowerCase()
  const i = n.lastIndexOf('.')
  return i === -1 ? '' : n.slice(i)
}

export default function App() {
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [resultUrl, setResultUrl] = useState(null)
  const [status, setStatus] = useState('idle') // idle | selected | processing | done | error
  const [error, setError] = useState('')
  const [metricsNote, setMetricsNote] = useState('')
  const [stagesApplied, setStagesApplied] = useState([])
  const [outputInfo, setOutputInfo] = useState(null)
  const [denoise, setDenoise] = useState(true)
  const [gamma, setGamma] = useState(1.4)
  const [bilateral, setBilateral] = useState(true)
  const [diffusion, setDiffusion] = useState(true)
  const [visibility, setVisibility] = useState(null)
  const [bgVideoOk, setBgVideoOk] = useState(true)
  const [terrain, setTerrain] = useState(null)
  const [terrainStatus, setTerrainStatus] = useState('idle') // idle | analyzing | done | error
  const [terrainError, setTerrainError] = useState('')
  const [tiTab, setTiTab] = useState('shadows') // shadows | hazards | route | legend
  const [tiLayer, setTiLayer] = useState('shadow') // none | shadow | hazard
  const [tiOpacity, setTiOpacity] = useState(70)
  const [tiCannyHi, setTiCannyHi] = useState(150)
  const [clickMode, setClickMode] = useState(null) // null | 'start' | 'target'
  const [routeStart, setRouteStart] = useState(null) // [nx, ny] normalized
  const [routeTarget, setRouteTarget] = useState(null)
  const [routeMode, setRouteMode] = useState('balanced')
  const [routeAlgo, setRouteAlgo] = useState('astar')
  const [routeResult, setRouteResult] = useState(null)
  const [routeStatus, setRouteStatus] = useState('idle') // idle | routing | done | error
  const [routeError, setRouteError] = useState('')
  const [routeUnderlay, setRouteUnderlay] = useState('hazard') // none | hazard | shadow
  const [tiSource, setTiSource] = useState('enhanced') // enhanced | upload
  const [tiFile, setTiFile] = useState(null)
  const [tiPreview, setTiPreview] = useState(null)
  const [libOpen, setLibOpen] = useState(false)
  const [libManifest, setLibManifest] = useState(null)
  const [libLoading, setLibLoading] = useState(false)
  const [libError, setLibError] = useState('')
  const [libSearch, setLibSearch] = useState('')
  const [libSort, setLibSort] = useState('title')
  const [libQuery, setLibQuery] = useState('all')
  const [libPage, setLibPage] = useState(0)
  const [libFailed, setLibFailed] = useState({})
  const [libSelected, setLibSelected] = useState(null)
  const [libProcessing, setLibProcessing] = useState(false)
  const [libProcError, setLibProcError] = useState('')
  const [, setLibVersion] = useState(0) // rerender when the memory cache fills
  const libCache = useRef({})
  const inputRef = useRef(null)
  const LIB_PAGE_SIZE = 24

  function clearTerrain() {
    setTerrain(null)
    setTerrainStatus('idle')
    setTerrainError('')
    setRouteStart(null)
    setRouteTarget(null)
    setRouteResult(null)
    setRouteStatus('idle')
    setRouteError('')
    setClickMode(null)
  }

  function pickFile(e) {
    const f = e.target.files?.[0]
    setError('')
    setMetricsNote('')
    setStagesApplied([])
    setOutputInfo(null)
    setVisibility(null)
    clearTerrain()
    setResultUrl((u) => {
      if (u && u.startsWith('blob:')) URL.revokeObjectURL(u)
      return null
    })
    if (!f) {
      setFile(null)
      setPreviewUrl((u) => {
        if (u) URL.revokeObjectURL(u)
        return null
      })
      setStatus('idle')
      return
    }
    if (!ALLOWED_EXTS.includes(extOf(f.name))) {
      setFile(null)
      setPreviewUrl(null)
      setStatus('error')
      setError(FORMAT_ERROR)
      return
    }
    if (f.size > MAX_BYTES) {
      setFile(null)
      setPreviewUrl(null)
      setStatus('error')
      setError('File exceeds 50 MB limit.')
      return
    }
    setFile(f)
    setPreviewUrl((u) => {
      if (u) URL.revokeObjectURL(u)
      return URL.createObjectURL(f)
    })
    setStatus('selected')
  }

  async function postEnhance(form) {
    try {
      return await fetch('/api/enhance', { method: 'POST', body: form })
    } catch (proxyErr) {
      // Dev fallback when Vite proxy is bypassed (e.g. preview build).
      const retry = new FormData()
      for (const [k, v] of form.entries()) retry.append(k, v)
      try {
        return await fetch('http://127.0.0.1:8000/api/enhance', {
          method: 'POST',
          body: retry,
        })
      } catch {
        throw proxyErr
      }
    }
  }

  async function process() {
    if (!file) return
    setStatus('processing')
    setError('')
    clearTerrain()
    try {
      const form = new FormData()
      // Explicit filename: without it some browsers send an empty filename
      // and the backend can no longer see the .jpg extension.
      form.append('file', file, file.name || 'upload.jpg')
      form.append('denoise', String(denoise))
      form.append('gamma', String(gamma))
      form.append('bilateral', String(bilateral))
      form.append('diffusion', String(diffusion))
      const res = await postEnhance(form)
      if (!res.ok) {
        let detail = ''
        try {
          const j = await res.json()
          detail = j.detail || ''
        } catch {
          /* empty error body */
        }
        setStatus('error')
        setError(detail || FORMAT_ERROR)
        return
      }
      const j = await res.json()
      setResultUrl(j.image)
      setMetricsNote(j.metrics_note || '')
      setStagesApplied(j.stages || [])
      setVisibility(j.visibility || null)
      setOutputInfo(j.output || null)
      setStatus('done')
    } catch {
      setStatus('error')
      setError(BACKEND_OFFLINE)
    }
  }

  function pickTiFile(e) {
    const f = e.target.files?.[0]
    setTiPreview((u) => {
      if (u) URL.revokeObjectURL(u)
      return null
    })
    setTiFile(null)
    if (!f) return
    if (!ALLOWED_EXTS.includes(extOf(f.name))) {
      setTerrainStatus('error')
      setTerrainError(FORMAT_ERROR)
      return
    }
    if (f.size > MAX_BYTES) {
      setTerrainStatus('error')
      setTerrainError('File exceeds 50 MB limit.')
      return
    }
    setTiFile(f)
    setTiPreview(URL.createObjectURL(f))
    clearTerrain()
  }

  async function runAnalyze() {
    if (terrainStatus === 'analyzing') return
    let blob = null
    let fname = 'image.png'
    if (tiSource === 'upload') {
      if (!tiFile) return
      blob = tiFile
      fname = tiFile.name || 'image.png'
    } else {
      if (!resultUrl) return
      blob = await (await fetch(resultUrl)).blob()
      fname = 'enhanced.png'
    }
    setTerrainStatus('analyzing')
    setTerrainError('')
    try {
      const form = new FormData()
      form.append('file', blob, fname)
      form.append('canny_hi', String(tiCannyHi))
      form.append('min_r', '12')
      form.append('max_r', '200')
      let res
      try {
        res = await fetch('/api/analyze', { method: 'POST', body: form })
      } catch (proxyErr) {
        const retry = new FormData()
        for (const [k, v] of form.entries()) retry.append(k, v)
        try {
          res = await fetch('http://127.0.0.1:8000/api/analyze', {
            method: 'POST',
            body: retry,
          })
        } catch {
          throw proxyErr
        }
      }
      if (!res.ok) {
        let detail = ''
        try {
          detail = (await res.json()).detail || ''
        } catch {
          /* empty error body */
        }
        setTerrainStatus('error')
        setTerrainError(detail || FORMAT_ERROR)
        return
      }
      setTerrain(await res.json())
      setRouteStart(null)
      setRouteTarget(null)
      setRouteResult(null)
      setRouteStatus('idle')
      setRouteError('')
      setTerrainStatus('done')
    } catch {
      setTerrainStatus('error')
      setTerrainError(BACKEND_OFFLINE)
    }
  }

  async function runRoute() {
    if (!terrain || !routeStart || !routeTarget || routeStatus === 'routing')
      return
    setRouteStatus('routing')
    setRouteError('')
    const body = JSON.stringify({
      analysis_id: terrain.analysis_id,
      start: routeStart,
      target: routeTarget,
      mode: routeMode,
      algorithm: routeAlgo,
    })
    try {
      let res
      try {
        res = await fetch('/api/route', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body,
        })
      } catch (proxyErr) {
        try {
          res = await fetch('http://127.0.0.1:8000/api/route', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body,
          })
        } catch {
          throw proxyErr
        }
      }
      const j = await res.json()
      if (!res.ok || !j.found) {
        setRouteStatus('error')
        setRouteError((j && (j.detail || j.reason)) || 'Route planning failed.')
        return
      }
      setRouteResult(j)
      setRouteStatus('done')
    } catch {
      setRouteStatus('error')
      setRouteError(BACKEND_OFFLINE)
    }
  }

  function selectTiTab(k) {
    setTiTab(k)
    // Each tab shows its own layer so the viewer always matches the numbers.
    if (k === 'shadows') setTiLayer('shadow')
    else if (k === 'hazards') setTiLayer('hazard')
    else setTiLayer('none')
  }

  function onViewerClick(e) {    if (!clickMode || tiTab !== 'route') return
    const rect = e.currentTarget.getBoundingClientRect()
    const nx = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width))
    const ny = Math.min(1, Math.max(0, (e.clientY - rect.top) / rect.height))
    const pt = [Math.round(nx * 10000) / 10000, Math.round(ny * 10000) / 10000]
    if (clickMode === 'start') setRouteStart(pt)
    else setRouteTarget(pt)
    setClickMode(null)
    setRouteResult(null)
    setRouteStatus('idle')
    setRouteError('')
  }

  function reset() {
    setFile(null)
    setError('')
    setMetricsNote('')
    setStagesApplied([])
    setOutputInfo(null)
    setVisibility(null)
    clearTerrain()
    setTiFile(null)
    setTiPreview((u) => {
      if (u) URL.revokeObjectURL(u)
      return null
    })
    setPreviewUrl((u) => {
      if (u) URL.revokeObjectURL(u)
      return null
    })
    setResultUrl((u) => {
      if (u && u.startsWith('blob:')) URL.revokeObjectURL(u)
      return null
    })
    setStatus('idle')
    if (inputRef.current) inputRef.current.value = ''
  }

  const analysisCopy = {
    idle: 'Upload a lunar image and reveal details hidden in the darkness.',
    selected: 'Your image is ready for processing.',
    processing: 'Uploading and validating your lunar image…',
    done: 'Your image was successfully processed by the backend.',
    error: error,
  }[status]

  async function openLibrary() {
    setLibOpen(true)
    setLibSelected(null)
    setLibProcError('')
    if (libManifest) return
    setLibLoading(true)
    setLibError('')
    try {
      let res
      try {
        res = await fetch('/api/library')
      } catch (proxyErr) {
        try {
          res = await fetch('http://127.0.0.1:8000/api/library')
        } catch {
          throw proxyErr
        }
      }
      if (!res.ok) throw new Error('bad status')
      setLibManifest(await res.json())
    } catch {
      setLibError(BACKEND_OFFLINE)
    } finally {
      setLibLoading(false)
    }
  }

  async function selectLibrary(item) {
    setLibSelected(item)
    setLibProcError('')
    if (libCache.current[item.id]) return // instant replay: no backend call
    setLibProcessing(true)
    try {
      const body = JSON.stringify({ id: item.id })
      let res
      try {
        res = await fetch('/api/library/enhance', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body,
        })
      } catch (proxyErr) {
        try {
          res = await fetch('http://127.0.0.1:8000/api/library/enhance', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body,
          })
        } catch {
          throw proxyErr
        }
      }
      if (!res.ok) {
        let detail = ''
        try {
          detail = (await res.json()).detail || ''
        } catch {
          /* empty error body */
        }
        setLibProcError(detail || FORMAT_ERROR)
        return
      }
      libCache.current[item.id] = await res.json()
      setLibVersion((v) => v + 1)
    } catch {
      setLibProcError(BACKEND_OFFLINE)
    } finally {
      setLibProcessing(false)
    }
  }

  const libNeedle = libSearch.trim().toLowerCase()
  const libFiltered = ((libManifest && libManifest.items) || [])
    .filter(
      (it) =>
        (libQuery === 'all' || it.query === libQuery) &&
        `${it.title} ${it.id}`.toLowerCase().includes(libNeedle),
    )
    .sort((a, b) =>
      libSort === 'query'
        ? a.query.localeCompare(b.query) || a.title.localeCompare(b.title)
        : a.title.localeCompare(b.title),
    )
  const libPages = Math.max(1, Math.ceil(libFiltered.length / LIB_PAGE_SIZE))
  const libPageItems = libFiltered.slice(
    libPage * LIB_PAGE_SIZE,
    (libPage + 1) * LIB_PAGE_SIZE,
  )
  const libPayload = libSelected ? libCache.current[libSelected.id] : null
  const tiBase = tiSource === 'upload' ? tiPreview : resultUrl

  return (
    <>
      {bgVideoOk && (
        <video
          className="bg-video"
          src="/deep-space-nebula-moewalls-com.mp4"
          autoPlay
          muted
          loop
          playsInline
          preload="auto"
          aria-hidden="true"
          tabIndex={-1}
          onError={() => setBgVideoOk(false)}
        />
      )}
      <div className="bg-overlay" aria-hidden="true" />
      <div className="page">
      <header className="topbar">
        <div className="brand">LUNARIS OBSERVATORY</div>
        <div className="top-meta">
          <span>MISSION CONTROL</span>
        </div>
      </header>

      <main>
        <section className="hero">
          <div className="hero-text">
            <p className="eyebrow">LIVE ORBITAL DATA</p>
            <h1>Explore the unknown.</h1>
            <p className="sub">
              Navigate the lunar surface, inspect permanently shadowed regions,
              and uncover what lies beyond the light.
            </p>
            <p className="sub ohrc">
              Demonstrating the OHRC (Chandrayaan-2, 0.25 m/pixel) enhancement
              approach of Barai et&nbsp;al., IJRPR 2025.
            </p>
          </div>

          <div className="moon-wrap" aria-label="Interactive 3D lunar map with markers">
            <Suspense fallback={<div className="placeholder">Loading 3D moon…</div>}>
              <Moon3D />
            </Suspense>
          </div>
        </section>

        <section className="cards centered">
          <article className="card analysis">
            <p className="card-label">FIELD MANUAL</p>
            <h2>New here? Start here.</h2>
            <p className="card-text">
              Five steps, top to bottom. Each section unlocks the next, so
              you never face everything at once.
            </p>
            <ol className="manual-steps">
              <li>
                <strong>Enhance.</strong> Upload a moon image (JPG / PNG /
                TIFF / WEBP, 50 MB max). The checkboxes are pipeline stages:
                denoisers clean grain, gamma lifts shadows. Press{' '}
                <em>Enhance image</em>.
              </li>
              <li>
                <strong>Visibility check.</strong> Appears after enhancing.
                Gray bars = before, cyan = after. Watch the left-edge spike
                melt into the middle — that is shadow becoming visible.
              </li>
              <li>
                <strong>Analyze terrain.</strong> Scroll to Terrain
                Intelligence. Use the enhanced result, or upload a clean
                image directly. Sliders tune detection; press{' '}
                <em>Analyze terrain</em>.
              </li>
              <li>
                <strong>Read the tabs.</strong> PSR analysis colors
                shadows (dark is <em>candidate</em>, never proven).
                Hazards paints risk green → red.
              </li>
              <li>
                <strong>Plan a route.</strong> In the Route tab press{' '}
                <em>Set start</em>, click the map, then <em>Set target</em>
                and click again. Pick Shortest, Safest or Balanced, then{' '}
                <em>Calculate route</em>.
              </li>
            </ol>
            <details className="manual-more">
              <summary>What do the numbers mean?</summary>
              <ul>
                <li>
                  <strong>Crushed-black %</strong> — pixels too dark to hold
                  detail, before → after. Falling means recovery.
                </li>
                <li>
                  <strong>Dark-region lift (×)</strong> — how many times
                  brighter the darkest quarter became.
                </li>
                <li>
                  <strong>Depth indicator</strong> — apparent relief
                  only. Never true depth; sizes are in pixels, not
                  meters.
                </li>
                <li>
                  <strong>Route cost / hazard</strong> — estimated from
                  image-derived costs. Unknown (gray) cells raise the cost.
                </li>
              </ul>
              <p>
                Tip: bright, detailed images give the best results. An
                already-black image holds little signal for anything to
                recover.
              </p>
            </details>
          </article>
        </section>

        <section className="cards centered">
          <article className="card analysis">
            <p className="card-label">IMAGE ANALYSIS</p>
            <h2>Enhance terrain</h2>
            <p className="card-text" role="status">
              {analysisCopy}
            </p>
            <input
              ref={inputRef}
              type="file"
              accept={ACCEPT}
              onChange={pickFile}
              className="file"
            />
            <fieldset className="stages">
              <legend>Pipeline stages (paper VII) — what each toggle does</legend>
              <label title="Non-Local Means: averages similar patches across the image to suppress low-light sensor noise">
                <input
                  type="checkbox"
                  checked={denoise}
                  onChange={(e) => setDenoise(e.target.checked)}
                />
                NLM (Non-Local Means) denoise — removes sensor noise
              </label>
              <label title="Bilateral filter: smooths flat areas but keeps crater edges sharp">
                <input
                  type="checkbox"
                  checked={bilateral}
                  onChange={(e) => setBilateral(e.target.checked)}
                />
                Bilateral smoothing (edge-preserving) — keeps edges sharp
              </label>
              <label title="Perona-Malik anisotropic diffusion: iterative smoothing that preserves edges. Different from generative diffusion models (future scope).">
                <input
                  type="checkbox"
                  checked={diffusion}
                  onChange={(e) => setDiffusion(e.target.checked)}
                />
                Anisotropic diffusion (Perona-Malik) — smooths noise, keeps structure
              </label>
              <label className="gamma" title="Gamma correction: brightens crushed shadows on the lightness channel">
                Gamma (brightness) correction
                <input
                  type="range"
                  min="0.5"
                  max="3"
                  step="0.1"
                  value={gamma}
                  onChange={(e) => setGamma(Number(e.target.value))}
                />
                <span>{gamma.toFixed(1)}</span>
              </label>
            </fieldset>
            <p className="hint">
              Always on: CLAHE (Contrast Limited Adaptive Histogram
              Equalization) on the lightness channel recovers contrast without
              shifting colours. The pipeline stretches faint existing signal —
              it does not invent terrain.
            </p>
            <div className="btn-row">
              <button
                type="button"
                onClick={process}
                disabled={!file || status === 'processing'}
              >
                {status === 'processing' ? 'Processing…' : 'Enhance image'}
              </button>
              <button type="button" className="ghost" onClick={openLibrary}>
                Image Library
              </button>
              {(file || resultUrl || status === 'error') && (
                <button
                  type="button"
                  className="ghost"
                  onClick={reset}
                  disabled={status === 'processing'}
                >
                  Reset
                </button>
              )}
            </div>
            <p className="hint">JPG / JPEG / PNG / TIFF / WEBP · 50 MB max</p>
          </article>
        </section>

        {(previewUrl || resultUrl) && (
          <section className="results">
            <div>
              <h3>Original</h3>
              {previewUrl && <img src={previewUrl} alt="Uploaded lunar image" />}
            </div>
            <div>
              <h3>Enhanced</h3>
              {resultUrl ? (
                <>
                  <img src={resultUrl} alt="Enhanced lunar terrain" />
                  {stagesApplied.length > 0 && (
                    <div className="chips" aria-label="Stages applied">
                      {stagesApplied.map((s) => (
                        <span key={s} className="chip">
                          {s}
                        </span>
                      ))}
                    </div>
                  )}
                  {metricsNote && <p className="note">{metricsNote}</p>}
                  {outputInfo?.downscaled && (
                    <p className="note">
                      Downscaled to {outputInfo.width}×{outputInfo.height} for
                      processing (cap {outputInfo.cap_px}px).
                    </p>
                  )}
                  <p className="disclaimer">
                    Enhanced for visibility — brightness/contrast stretched. Not
                    a scientific measurement of ice.
                  </p>
                  <a href={resultUrl} download="lunaris-enhanced.png">
                    Download PNG
                  </a>
                </>
              ) : (
                <div className="placeholder">
                  {status === 'processing'
                    ? 'Uploading and validating your lunar image…'
                    : 'Enhanced output will appear here.'}
                </div>
              )}
            </div>
          </section>
        )}

        {resultUrl && visibility && (
          <section className="cards centered">
            <article className="card analysis">
              <p className="card-label">VISIBILITY CHECK</p>
              <h2>How much more can you see?</h2>
              <p className="card-text">
                Brightness histograms of the original versus the enhanced
                image, plus what changed in the shadows.
              </p>
              <VisHistogram
                before={visibility.hist_before}
                after={visibility.hist_after}
              />
              <div className="vis-legend" aria-hidden="true">
                <span>
                  <i style={{ background: '#5b6b82' }} />
                  before
                </span>
                <span>
                  <i style={{ background: '#7dd3fc' }} />
                  after
                </span>
                <span>dark → bright</span>
              </div>
              <div className="vis-tiles">
                <div className="vis-tile">
                  <span className="vis-num">
                    {visibility.crushed_before}% → {visibility.crushed_after}%
                  </span>
                  <span className="vis-label">
                    Crushed-black pixels recovered
                  </span>
                </div>
                <div className="vis-tile">
                  <span className="vis-num">{visibility.dark_lift}×</span>
                  <span className="vis-label">
                    Dark-region brightness lift
                  </span>
                </div>
                <div className="vis-tile">
                  <span className="vis-num">{visibility.contrast_gain}×</span>
                  <span className="vis-label">Contrast gain</span>
                </div>
              </div>
            </article>
          </section>
        )}

        {
          <section className="cards centered">
            <article className="card analysis ti-wide">
              <p className="card-label">TERRAIN INTELLIGENCE</p>
              <h2>Classical shadow, hazard & route analysis</h2>
              <p className="card-text">
                No AI — deterministic OpenCV geometry on your enhanced
                image, or on a clean image you upload directly. Dark means
                observable shadow (never proven PSR), and every depth
                number is an apparent indicator, not true depth.
              </p>
              <div className="ti-row">
                <span className="hint">Source image:</span>
                <div className="btn-row">
                  <button
                    type="button"
                    className={tiSource === 'enhanced' ? '' : 'ghost'}
                    onClick={() => {
                      setTiSource('enhanced')
                      clearTerrain()
                    }}
                    disabled={!resultUrl}
                  >
                    Enhanced result
                  </button>
                  <button
                    type="button"
                    className={tiSource === 'upload' ? '' : 'ghost'}
                    onClick={() => {
                      setTiSource('upload')
                      clearTerrain()
                    }}
                  >
                    Upload clean image
                  </button>
                </div>
              </div>
              {tiSource === 'upload' && (
                <>
                  <input
                    type="file"
                    accept={ACCEPT}
                    onChange={pickTiFile}
                    className="file"
                    aria-label="Clean image for direct terrain analysis"
                  />
                  {tiFile && (
                    <p className="hint">Direct image: {tiFile.name}</p>
                  )}
                </>
              )}
              <div className="btn-row">
                <button
                  type="button"
                  onClick={runAnalyze}
                  disabled={
                    terrainStatus === 'analyzing' ||
                    (tiSource === 'upload' ? !tiFile : !resultUrl)
                  }
                >
                  {terrainStatus === 'analyzing'
                    ? 'Analyzing…'
                    : terrain
                      ? 'Re-analyze terrain'
                      : 'Analyze terrain'}
                </button>
              </div>
              <div className="ti-sliders">
                <label className="ti-slider">
                  Edge sensitivity (Canny high)
                  <input
                    type="range"
                    min="80"
                    max="300"
                    step="5"
                    value={tiCannyHi}
                    onChange={(e) => setTiCannyHi(Number(e.target.value))}
                  />
                  <span>{tiCannyHi}</span>
                </label>
              </div>
              <p className="hint">
                Sliders apply on re-analyze. Analysis runs capped at 960px;
                coordinates scale back automatically.
              </p>
              {terrainError && (
                <p className="note" role="alert">
                  {terrainError}
                </p>
              )}
              {terrainStatus === 'analyzing' && (
                <p className="note" role="status">
                  Segmenting shadows, mapping hazards…
                </p>
              )}
              {terrain && (
                <>
                  <div className="ti-tabs" role="tablist" aria-label="Terrain modules">
                    {[
                      ['shadows', 'PSR analysis'],
                      ['hazards', 'Hazards'],
                      ['route', 'Route'],
                      ['legend', 'Legend'],
                    ].map(([k, label]) => (
                      <button
                        key={k}
                        type="button"
                        role="tab"
                        aria-selected={tiTab === k}
                        className={tiTab === k ? 'is-active' : ''}
                        onClick={() => selectTiTab(k)}
                      >
                        {label}
                      </button>
                    ))}
                  </div>
                  {tiTab === 'route' ? (
                    <>
                    <div className="ti-row">
                      <label className="hint" htmlFor="ti-route-layer">
                        Underlay
                      </label>
                      <select
                        id="ti-route-layer"
                        className="ti-select"
                        value={routeUnderlay}
                        onChange={(e) => setRouteUnderlay(e.target.value)}
                      >
                        <option value="none">None (base image)</option>
                        <option value="hazard">Hazard map</option>
                        <option value="shadow">
                          Shadow / PSR candidates
                        </option>
                      </select>
                      {routeUnderlay !== 'none' && (
                        <label className="ti-slider ti-op">
                          Opacity
                          <input
                            type="range"
                            min="0"
                            max="100"
                            value={tiOpacity}
                            onChange={(e) =>
                              setTiOpacity(Number(e.target.value))
                            }
                          />
                          <span>{tiOpacity}%</span>
                        </label>
                      )}
                    </div>
                    <p className="hint">
                      Route draws on top — red hazard / dark shadow shows
                      what the optimal path avoids.
                    </p>
                    </>
                  ) : (
                    <div className="ti-row">
                      <label className="hint" htmlFor="ti-layer">
                        Overlay
                      </label>
                      <select
                        id="ti-layer"
                        className="ti-select"
                        value={tiLayer}
                        onChange={(e) => setTiLayer(e.target.value)}
                      >
                        <option value="none">None (base image)</option>
                        {tiTab === 'shadows' && (
                          <option value="shadow">
                            Shadow / PSR candidates
                          </option>
                        )}
                        {tiTab === 'hazards' && (
                          <option value="hazard">Hazard map</option>
                        )}
                      </select>
                      {tiLayer !== 'none' && (
                        <label className="ti-slider ti-op">
                          Opacity
                          <input
                            type="range"
                            min="0"
                            max="100"
                            value={tiOpacity}
                            onChange={(e) =>
                              setTiOpacity(Number(e.target.value))
                            }
                          />
                          <span>{tiOpacity}%</span>
                        </label>
                      )}
                    </div>
                  )}
                  <div
                    className="ti-viewer"
                    onClick={onViewerClick}
                    style={clickMode ? { cursor: 'crosshair' } : undefined}
                  >
                    <img src={tiBase} alt="Terrain under analysis" />
                    {tiTab === 'route' ? (
                      <>
                        {routeUnderlay !== 'none' && (
                          <img
                            src={terrain.overlays[routeUnderlay]}
                            alt=""
                            className="ti-overlay"
                            style={{ opacity: tiOpacity / 100 }}
                            aria-hidden="true"
                          />
                        )}
                        {routeResult && (
                          <RoutePath
                            result={routeResult}
                            dims={terrain.dims}
                          />
                        )}
                      </>
                    ) : (
                      tiLayer !== 'none' && (
                        <img
                          src={terrain.overlays[tiLayer]}
                          alt=""
                          className="ti-overlay"
                          style={{ opacity: tiOpacity / 100 }}
                          aria-hidden="true"
                        />
                      )
                    )}
                    {routeStart && (
                      <span
                        className="ti-marker ti-start"
                        style={{
                          left: `${routeStart[0] * 100}%`,
                          top: `${routeStart[1] * 100}%`,
                        }}
                      />
                    )}
                    {routeTarget && (
                      <span
                        className="ti-marker ti-target"
                        style={{
                          left: `${routeTarget[0] * 100}%`,
                          top: `${routeTarget[1] * 100}%`,
                        }}
                      />
                    )}
                  </div>
                  {tiTab === 'shadows' && (
                    <>
                      <LegendStrip group="shadow" />
                      <AlgoDetails tab="shadows" />
                      <div className="vis-tiles">
                        <div className="vis-tile">
                          <span className="vis-num">
                            {terrain.shadow.shadow_pct}%
                          </span>
                          <span className="vis-label">Shadowed terrain</span>
                        </div>
                        <div className="vis-tile">
                          <span className="vis-num">
                            {terrain.shadow.transition_pct}%
                          </span>
                          <span className="vis-label">Transition zone</span>
                        </div>
                        <div className="vis-tile">
                          <span className="vis-num">
                            {terrain.shadow.otsu_threshold}
                          </span>
                          <span className="vis-label">Otsu threshold</span>
                        </div>
                      </div>
                      {terrain.shadow.regions.length > 0 && (
                        <table className="method">
                          <thead>
                            <tr>
                              <th scope="col">Region</th>
                              <th scope="col">Area %</th>
                              <th scope="col">Compact.</th>
                              <th scope="col">Class</th>
                            </tr>
                          </thead>
                          <tbody>
                            {terrain.shadow.regions.slice(0, 8).map((r) => (
                              <tr key={r.id}>
                                <td>PSR?{r.id}</td>
                                <td>{r.area_pct}</td>
                                <td>{r.compactness}</td>
                                <td>{r.class}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      )}
                      <p className="note">
                        Green = illuminated, yellow = transition, dark =
                        shadow, orange = candidate boundary. Dark is an
                        observable shadow region — permanence is UNKNOWN
                        without illumination data.
                      </p>
                    </>
                  )}
                  {tiTab === 'hazards' && (
                    <>
                      <LegendStrip group="hazard" />
                      <AlgoDetails tab="hazards" />
                      <div className="vis-tiles">
                        {Object.entries(terrain.hazard.class_pct).map(
                          ([k, v]) => (
                            <div className="vis-tile" key={k}>
                              <span className="vis-num">{v}%</span>
                              <span className="vis-label">{k}</span>
                            </div>
                          ),
                        )}
                      </div>
                      <div className="chips" aria-label="Hazard weights">
                        {Object.entries(terrain.hazard.weights).map(
                          ([k, v]) => (
                            <span key={k} className="chip">
                              {k} {v}
                            </span>
                          ),
                        )}
                      </div>
                      <p className="note">
                        Mean score {terrain.hazard.mean_score}; contributions —
                        slope {terrain.hazard.mean_contrib.slope}, roughness{' '}
                        {terrain.hazard.mean_contrib.roughness}, boundary{' '}
                        {terrain.hazard.mean_contrib.crater_boundary},
                        uncertainty{' '}
                        {terrain.hazard.mean_contrib.uncertainty}. Green = low,
                        yellow = moderate, red = high, gray = unknown
                        (shadowed and featureless).
                      </p>
                    </>
                  )}
                  {tiTab === 'route' && (
                    <>
                      <LegendStrip group="route" />
                      <AlgoDetails tab="route" />
                      <div className="ti-row">
                        <div className="btn-row">
                          <button
                            type="button"
                            className={clickMode === 'start' ? '' : 'ghost'}
                            onClick={() =>
                              setClickMode(
                                clickMode === 'start' ? null : 'start',
                              )
                            }
                          >
                            Set start{routeStart ? ' ✓' : ''}
                          </button>
                          <button
                            type="button"
                            className={clickMode === 'target' ? '' : 'ghost'}
                            onClick={() =>
                              setClickMode(
                                clickMode === 'target' ? null : 'target',
                              )
                            }
                          >
                            Set target{routeTarget ? ' ✓' : ''}
                          </button>
                          <button
                            type="button"
                            className="ghost"
                            onClick={() => {
                              setRouteStart(null)
                              setRouteTarget(null)
                              setRouteResult(null)
                              setRouteStatus('idle')
                              setRouteError('')
                            }}
                          >
                            Clear
                          </button>
                        </div>
                      </div>
                      <p className="hint">
                        {!routeStart || !routeTarget
                          ? 'Pick “Set start”, click the map, then “Set target” and click again.'
                          : `Start (${routeStart[0]}, ${routeStart[1]}) → target (${routeTarget[0]}, ${routeTarget[1]}), normalized.`}
                      </p>
                      <div className="ti-row">
                        <select
                          className="ti-select"
                          value={routeMode}
                          onChange={(e) => setRouteMode(e.target.value)}
                          aria-label="Route mode"
                        >
                          <option value="shortest">Shortest</option>
                          <option value="safest">Safest</option>
                          <option value="balanced">Balanced</option>
                        </select>
                        <select
                          className="ti-select"
                          value={routeAlgo}
                          onChange={(e) => setRouteAlgo(e.target.value)}
                          aria-label="Pathfinding algorithm"
                        >
                          <option value="astar">A*</option>
                          <option value="dijkstra">Dijkstra</option>
                        </select>
                        <div className="btn-row">
                          <button
                            type="button"
                            onClick={runRoute}
                            disabled={
                              !routeStart ||
                              !routeTarget ||
                              routeStatus === 'routing'
                            }
                          >
                            {routeStatus === 'routing'
                              ? 'Calculating…'
                              : 'Calculate route'}
                          </button>
                        </div>
                      </div>
                      {routeError && (
                        <p className="note" role="alert">
                          {routeError}
                        </p>
                      )}
                      {routeStatus === 'routing' && (
                        <p className="note" role="status">
                          Searching the cost grid…
                        </p>
                      )}
                      {routeResult && (
                        <>
                          <div className="vis-tiles">
                            <div className="vis-tile">
                              <span className="vis-num">
                                {routeResult.distance_px} px
                              </span>
                              <span className="vis-label">Route distance</span>
                            </div>
                            <div className="vis-tile">
                              <span className="vis-num">
                                {routeResult.total_cost}
                              </span>
                              <span className="vis-label">
                                Total cost ({routeResult.algorithm},{' '}
                                {routeResult.mode})
                              </span>
                            </div>
                            <div className="vis-tile">
                              <span className="vis-num">
                                {routeResult.max_hazard}
                              </span>
                              <span className="vis-label">
                                Max hazard · {routeResult.unknown_pct}%
                                unknown
                              </span>
                            </div>
                          </div>
                          <p className="note">
                            Avg hazard score {routeResult.avg_hazard_score},{' '}
                            {routeResult.path_len_cells} cells,{' '}
                            {routeResult.unknown_cells} unknown cells. Route
                            is ESTIMATED from image-derived costs — blue =
                            start, red cross = target.
                          </p>
                        </>
                      )}
                    </>
                  )}
                  {tiTab === 'legend' && (
                    <>
                      <div className="legend-grid">
                        {Object.values(TI_LEGENDS).map((g) => (
                          <div className="legend-group" key={g.title}>
                            <h4>{g.title}</h4>
                            <ul>
                              {g.items.map(([color, text]) => (
                                <li key={text}>
                                  <i style={{ background: color }} />
                                  {text}
                                </li>
                              ))}
                            </ul>
                          </div>
                        ))}
                      </div>
                      <p className="note">
                        Same colors as the overlays — what you see in the
                        viewer is what each swatch means.
                      </p>
                    </>
                  )}
                  <div className="ti-banner" role="note">
                    {terrain.no_hallucination}
                  </div>
                  <h3 className="mini-head">Evidence legend</h3>
                  <ul className="glossary">
                    {terrain.evidence_legend.map((e) => (
                      <li key={e.tag}>
                        <strong>{e.tag}</strong> — {e.meaning}
                      </li>
                    ))}
                  </ul>
                </>
              )}
            </article>
          </section>
        }

        <section className="info">
          <article>
            <p className="card-label">WHAT IS A PSR?</p>
            <h2>Permanently shadowed regions</h2>
            <p>
              The Moon&apos;s axis is tilted only ~1.5°, so near the lunar
              north and south poles the Sun always stays near the horizon. Deep
              polar crater floors and rims never receive direct sunlight.
            </p>
            <p>
              These shadowed floors are among the coldest places in the solar
              system, often below ~40–100 K (the referenced paper cites traps
              near ~−250 °C). The cold traps preserve water ice and volatiles
              delivered by comets and asteroids over billions of years.
            </p>
            <ul>
              <li>Science: a pristine record of solar system volatiles.</li>
              <li>
                Exploration: water ice means drinking water, air, and rocket
                fuel (ISRU — In-Situ Resource Utilization) for Artemis-era
                missions.
              </li>
              <li>
                LCROSS (Lunar Crater Observation and Sensing Satellite, 2009,
                Cabeus crater) excavated and confirmed water in a PSR
                (Permanently Shadowed Region).
              </li>
            </ul>
            <p className="instruments">
              Imaged at 0.25 m/pixel by the OHRC (Orbiter High-Resolution
              Camera) aboard ISRO&apos;s Chandrayaan-2 — the data source for
              the referenced paper — alongside the LRO (Lunar Reconnaissance
              Orbiter: LROC NAC — Lunar Reconnaissance Orbiter Camera Narrow
              Angle Camera, LOLA — Lunar Orbiter Laser Altimeter, Diviner —
              Diviner Lunar Radiometer), Chandrayaan-1/2, and ShadowCam, which
              images PSRs using faint secondary light.
            </p>
          </article>

          <article>
            <p className="card-label">WHY IMAGES ARE SO HARD</p>
            <h2>Signal buried in darkness</h2>
            <p>
              Direct illumination inside a PSR (Permanently Shadowed Region)
              can be ~0% — almost no direct sun. What little light exists is
              secondary: earthshine, starlight, and sunlight scattered off
              bright crater rims, so the signal is extremely faint.
            </p>
            <p>
              Raw OHRC (Orbiter High-Resolution Camera) frames look black: low
              photon counts, heavy sensor noise, crushed shadows, almost no
              contrast, and a poor SNR (Signal-to-Noise Ratio). Standard
              histogram equalization and contrast stretching fail here — they
              ignore the lighting and SNR regime, and aggressive settings
              invent artifacts.
            </p>
            <p>
              The pipeline therefore denoises with NLM (Non-Local Means),
              recovers contrast with CLAHE (Contrast Limited Adaptive Histogram
              Equalization) on the lightness channel plus gamma (brightness)
              correction, smooths while keeping edges (bilateral filter), and
              diffuses noise while preserving edges with anisotropic diffusion
              (Perona-Malik). It stretches existing faint signal to reveal
              faint terrain — it does not invent terrain.
            </p>
          </article>
        </section>

        <section className="research">
          <article>
            <p className="card-label">RESEARCH BASIS</p>
            <h2>Paper methodology (Table 1)</h2>
            <table className="method">
              <thead>
                <tr>
                  <th scope="col">Stage</th>
                  <th scope="col">Technique</th>
                  <th scope="col">Purpose</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>Preprocessing</td>
                  <td>
                    NLM (Non-Local Means) denoise, CLAHE (Contrast Limited
                    Adaptive Histogram Equalization), gamma (brightness)
                    correction
                  </td>
                  <td>Suppress low-light noise; recover contrast</td>
                </tr>
                <tr>
                  <td>Enhancement</td>
                  <td>
                    Adaptive contrast stretch, bilateral filter
                    (edge-preserving smoothing)
                  </td>
                  <td>Reveal shadowed areas; preserve crater edges</td>
                </tr>
                <tr>
                  <td>Post-processing</td>
                  <td>
                    Anisotropic diffusion (Perona-Malik), edge-preserving
                    denoise
                  </td>
                  <td>Smooth noise while keeping sharp structure</td>
                </tr>
                <tr>
                  <td>Evaluation</td>
                  <td>
                    SNR (Signal-to-Noise Ratio), PSNR (Peak Signal-to-Noise
                    Ratio), SSIM (Structural Similarity Index), FVI (Feature
                    Visibility Index)
                  </td>
                  <td>Objective before/after comparison</td>
                </tr>
              </tbody>
            </table>
            <p className="cite">
              Based on Barai, Nirmal, Negi, Kamble &amp; Hirave (2025),
              “Enhancement of Permanently Shadowed Regions (PSR) of Lunar
              Craters Captured by OHRC of Chandrayaan-2”, IJRPR Vol 6(4), pp
              7751–7758.{' '}
              <a
                href="https://ijrpr.com/uploads/V6ISSUE4/IJRPR42538.pdf"
                target="_blank"
                rel="noreferrer"
              >
                Read the paper (PDF)
              </a>
            </p>
          </article>

          <article>
            <p className="card-label">APPLICATIONS</p>
            <h2>What clearer images enable</h2>
            <ul>
              <li>
                Landing-site selection: clearer hazard mapping and terrain
                evaluation for safer touchdown zones.
              </li>
              <li>
                Water-ice candidates: improved visibility helps researchers
                shortlist shadowed regions for follow-up study — this demo
                does not itself detect ice.
              </li>
              <li>
                Lunar morphology: crater depth, ridges, rock formations, and
                surface textures become measurable.
              </li>
            </ul>
            <h3 className="mini-head">What this website already does</h3>
            <ul className="glossary">
              <li>
                <strong>NLM (Non-Local Means) denoise</strong> — averages
                similar patches to remove low-light sensor noise before
                contrast is stretched.
              </li>
              <li>
                <strong>
                  CLAHE (Contrast Limited Adaptive Histogram Equalization)
                </strong>{' '}
                — local contrast recovery on the lightness channel, so colours
                don&apos;t shift.
              </li>
              <li>
                <strong>Gamma (brightness) correction</strong> — lifts crushed
                shadows; adjustable 0.5–3.0 in the panel above.
              </li>
              <li>
                <strong>Bilateral smoothing</strong> — edge-preserving
                smoothing that keeps crater rims sharp.
              </li>
              <li>
                <strong>Anisotropic diffusion (Perona-Malik)</strong> —
                iterative smoothing that flattens noisy flats but stops at
                edges. Not a generative diffusion model.
              </li>
              <li>
                <strong>
                  SNR (Signal-to-Noise Ratio), PSNR (Peak Signal-to-Noise
                  Ratio), SSIM (Structural Similarity Index), FVI (Feature
                  Visibility Index)
                </strong>{' '}
                — objective before/after scores shown under every result.
              </li>
            </ul>
            <h3 className="mini-head">Future scope per the paper (not in this demo)</h3>
            <ul className="glossary">
              <li>
                <strong>
                  SRGAN (Super-Resolution Generative Adversarial Network)
                </strong>{' '}
                — a GAN (Generative Adversarial Network) / CNN (Convolutional
                Neural Network) model to upscale low-resolution PSR
                (Permanently Shadowed Region) images and reconstruct missing
                detail.
              </li>
              <li>
                <strong>Adaptive AI (Artificial Intelligence)-driven enhancement</strong>{' '}
                — ML (Machine Learning) that auto-tunes contrast, denoising,
                and feature extraction per image instead of manual sliders.
              </li>
              <li>
                <strong>Diffusion-model refinement</strong> — generative
                diffusion models that iteratively denoise while preserving fine
                geological detail; different from the Perona-Malik anisotropic
                diffusion filter already used here.
              </li>
              <li>
                <strong>Automated batch pipelines + OHRC (Orbiter High-Resolution Camera) scale</strong>{' '}
                — hands-free processing of large Chandrayaan-2 datasets and
                real-time onboard enhancement for lunar missions, with
                SNR/PSNR/SSIM/FVI comparison against histogram equalization,
                CLAHE, and deep-learning methods.
              </li>
            </ul>
          </article>
        </section>
      </main>

      <footer>
        <span>© 2026 LUNARIS OBSERVATORY</span>
        <span>MISSION CONTROL</span>
        <span className="ok">SYSTEM OPERATIONAL</span>
      </footer>

      {libOpen && (
        <div
          className="lib-overlay"
          role="dialog"
          aria-modal="true"
          aria-label="Image Library"
          onClick={() => setLibOpen(false)}
        >
          <div className="lib-modal" onClick={(e) => e.stopPropagation()}>
            <div className="lib-head">
              <div>
                <p className="card-label">IMAGE LIBRARY</p>
                <h2>
                  {libSelected
                    ? libSelected.title
                    : 'Browse shadowed-terrain frames'}
                </h2>
                <p className="lib-sub">
                  {libManifest
                    ? `${libManifest.count ?? libFiltered.length} public-domain frames · queries derived from Barai et al. 2025`
                    : 'Public-domain lunar frames, ready to present.'}
                </p>
              </div>
              <button
                type="button"
                className="ghost"
                onClick={() => setLibOpen(false)}
              >
                Close
              </button>
            </div>

            {!libSelected && (
              <>
                <div className="lib-toolbar">
                  <input
                    type="search"
                    placeholder="Search titles or IDs…"
                    value={libSearch}
                    onChange={(e) => {
                      setLibSearch(e.target.value)
                      setLibPage(0)
                    }}
                    aria-label="Search library"
                  />
                  <select
                    value={libQuery}
                    onChange={(e) => {
                      setLibQuery(e.target.value)
                      setLibPage(0)
                    }}
                    aria-label="Filter by paper query"
                  >
                    <option value="all">All queries</option>
                    {(libManifest?.queries || []).map((qq) => (
                      <option key={qq.q} value={qq.q}>
                        {qq.q}
                      </option>
                    ))}
                  </select>
                  <select
                    value={libSort}
                    onChange={(e) => setLibSort(e.target.value)}
                    aria-label="Sort library"
                  >
                    <option value="title">Title A–Z</option>
                    <option value="query">Group by query</option>
                  </select>
                </div>

                {libLoading && (
                  <div className="placeholder">Loading library manifest…</div>
                )}
                {libError && (
                  <p className="card-text" role="alert">
                    {libError}
                  </p>
                )}
                {!libLoading && !libError && libFiltered.length === 0 && (
                  <div className="placeholder">
                    No frames match this search.
                  </div>
                )}
                <div className="lib-grid">
                  {libPageItems.map((it) =>
                    libFailed[it.id] ? (
                      <div key={it.id} className="lib-card lib-broken">
                        <span>Frame unavailable</span>
                        <small>{it.title}</small>
                      </div>
                    ) : (
                      <button
                        key={it.id}
                        type="button"
                        className="lib-card"
                        onClick={() => selectLibrary(it)}
                        title={it.title}
                      >
                        <img
                          src={it.thumb}
                          alt={it.title}
                          loading="lazy"
                          onError={() =>
                            setLibFailed((f) => ({ ...f, [it.id]: true }))
                          }
                        />
                        <span className="lib-title">{it.title}</span>
                        <small>{it.query}</small>
                      </button>
                    ),
                  )}
                </div>
                <div className="lib-pager">
                  <button
                    type="button"
                    className="ghost"
                    disabled={libPage === 0}
                    onClick={() => setLibPage((p) => Math.max(0, p - 1))}
                  >
                    ← Prev
                  </button>
                  <span>
                    Page {libPage + 1} of {libPages} · {libFiltered.length}{' '}
                    frames
                  </span>
                  <button
                    type="button"
                    className="ghost"
                    disabled={libPage + 1 >= libPages}
                    onClick={() =>
                      setLibPage((p) => Math.min(libPages - 1, p + 1))
                    }
                  >
                    Next →
                  </button>
                </div>
                <p className="hint">
                  All frames: {libManifest?.attribution || 'Public domain'}.
                  Curated to moon craters and dark regions only — every frame
                  confirmed against its NASA record.
                </p>
              </>
            )}

            {libSelected && (
              <div className="lib-compare">
                <button
                  type="button"
                  className="ghost"
                  onClick={() => setLibSelected(null)}
                >
                  ← Back to grid
                </button>
                {libProcError && (
                  <p className="card-text" role="alert">
                    {libProcError}
                  </p>
                )}
                {libProcessing && !libPayload && (
                  <div className="placeholder">
                    Enhancing once with default pipeline settings — caching for
                    instant replay…
                  </div>
                )}
                {libPayload && (
                  <>
                    <div className="compare-grid">
                      <div>
                        <h3>Original</h3>
                        <img
                          src={libSelected.full}
                          alt={`${libSelected.title} (original)`}
                        />
                      </div>
                      <div>
                        <h3>Enhanced</h3>
                        <img
                          src={libPayload.image}
                          alt={`${libSelected.title} (enhanced)`}
                        />
                      </div>
                    </div>
                    <div className="chips" aria-label="Stages applied">
                      {(libPayload.stages || []).map((s) => (
                        <span key={s} className="chip">
                          {s}
                        </span>
                      ))}
                    </div>
                    {libPayload.metrics && (
                      <table className="metrics">
                        <caption>
                          Evaluation (paper VII.D) — luminance · default
                          settings
                        </caption>
                        <tbody>
                          <tr>
                            <th scope="row">
                              SNR (Signal-to-Noise Ratio) before → after
                            </th>
                            <td>
                              {libPayload.metrics.snr_before_db} →{' '}
                              {libPayload.metrics.snr_after_db} dB
                            </td>
                          </tr>
                          <tr>
                            <th scope="row">
                              PSNR (Peak Signal-to-Noise Ratio)
                            </th>
                            <td>{libPayload.metrics.psnr_db} dB</td>
                          </tr>
                          <tr>
                            <th scope="row">
                              SSIM (Structural Similarity Index)
                            </th>
                            <td>{libPayload.metrics.ssim}</td>
                          </tr>
                          <tr>
                            <th scope="row">
                              FVI (Feature Visibility Index, proxy)
                            </th>
                            <td>{libPayload.metrics.fvi_proxy}</td>
                          </tr>
                        </tbody>
                      </table>
                    )}
                    {libPayload.metrics_note && (
                      <p className="note">{libPayload.metrics_note}</p>
                    )}
                    <p className="disclaimer">
                      Enhanced for visibility — brightness/contrast stretched.
                      Not a scientific measurement of ice. ·{' '}
                      {libSelected.attribution}
                    </p>
                    <div className="btn-row">
                      <a
                        className="btn-link"
                        href={`/api/library/compare?id=${encodeURIComponent(libSelected.id)}`}
                        download={`lunaris-${libSelected.id}-compare.png`}
                      >
                        Export comparison PNG
                      </a>
                      <a
                        className="btn-link ghost-link"
                        href={libPayload.image}
                        download={`lunaris-${libSelected.id}-enhanced.png`}
                      >
                        Download enhanced
                      </a>
                    </div>
                  </>
                )}
              </div>
            )}
          </div>
        </div>
      )}
      </div>
    </>
  )
}
