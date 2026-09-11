import { useRef, useState } from 'react'
import './App.css'

const ACCEPT = '.jpg,.jpeg,.png,.tif,.tiff,.webp'
const ALLOWED_EXTS = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp']
const MAX_BYTES = 50 * 1024 * 1024
const BACKEND_OFFLINE =
  'Unable to connect to the backend. Please check whether FastAPI is running.'
const FORMAT_ERROR =
  'Unsupported format. Please upload JPG, JPEG, PNG, TIFF, or WEBP (50 MB max).'

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
  const [metrics, setMetrics] = useState(null)
  const [metricsNote, setMetricsNote] = useState('')
  const [stagesApplied, setStagesApplied] = useState([])
  const [outputInfo, setOutputInfo] = useState(null)
  const [denoise, setDenoise] = useState(true)
  const [gamma, setGamma] = useState(1.4)
  const [bilateral, setBilateral] = useState(true)
  const [diffusion, setDiffusion] = useState(true)
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

  function pickFile(e) {
    const f = e.target.files?.[0]
    setError('')
    setMetrics(null)
    setStagesApplied([])
    setOutputInfo(null)
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
      setMetrics(j.metrics || null)
      setMetricsNote(j.metrics_note || '')
      setStagesApplied(j.stages || [])
      setOutputInfo(j.output || null)
      setStatus('done')
    } catch {
      setStatus('error')
      setError(BACKEND_OFFLINE)
    }
  }

  function reset() {
    setFile(null)
    setError('')
    setMetrics(null)
    setMetricsNote('')
    setStagesApplied([])
    setOutputInfo(null)
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

  return (
    <div className="page">
      <header className="topbar">
        <div className="brand">LUNARIS OBSERVATORY</div>
        <div className="top-meta">
          <span>MISSION CONTROL</span>
          <span className="dot" aria-hidden="true" />
          <span className="ok">SYSTEM OPERATIONAL</span>
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

          <div className="moon-wrap" aria-label="Demo lunar map with markers">
            <div className="moon">
              <div className="crater c1" />
              <div className="crater c2" />
              <div className="crater c3" />
              <div className="crater c4" />
              <div className="terminator" />
              <span className="marker m1">
                <i /> PSR-01
              </span>
              <span className="marker m2">
                <i /> CRATER 04
              </span>
              <span className="marker m3">
                <i /> SHADOW ZONE
              </span>
            </div>
          </div>
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
                  {metrics && (
                    <table className="metrics">
                      <caption>Evaluation (paper VII.D) — luminance</caption>
                      <tbody>
                        <tr>
                          <th scope="row">
                            SNR (Signal-to-Noise Ratio) before → after
                          </th>
                          <td>
                            {metrics.snr_before_db} → {metrics.snr_after_db} dB
                          </td>
                        </tr>
                        <tr>
                          <th scope="row">PSNR (Peak Signal-to-Noise Ratio)</th>
                          <td>{metrics.psnr_db} dB</td>
                        </tr>
                        <tr>
                          <th scope="row">
                            SSIM (Structural Similarity Index)
                          </th>
                          <td>{metrics.ssim}</td>
                        </tr>
                        <tr>
                          <th scope="row">
                            FVI (Feature Visibility Index, proxy)
                          </th>
                          <td>{metrics.fvi_proxy}</td>
                        </tr>
                      </tbody>
                    </table>
                    <p className="note">
                      Higher SNR / PSNR means cleaner signal; SSIM near 1.0
                      means structure was preserved; FVI proxy above 1.0 means
                      contrast visibility improved.
                    </p>
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
  )
}
