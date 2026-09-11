# Image Library — implementation notes (Deliverable 3)

## Where caching is stored
- **Frontend (per session, zero HTTP):** `libCache` (`useRef` map in
  `frontend/src/App.jsx`), keyed by library id. Reopening an image in the
  same session renders instantly — no backend call.
- **Backend (shared across sessions/users, zero reprocessing):**
  `backend/.cache/library/<sha256>/` containing `original` (fetched bytes),
  `enhanced.png`, `compare.png`, `meta.json` (payload with metrics/stages).
  Key = SHA-256 of `PIPELINE_VERSION + image id + default settings`, so a
  future pipeline change (bump `PIPELINE_VERSION` in `backend/library.py`)
  invalidates old entries automatically. This directory is git-ignored;
  the served snapshot `backend/library_manifest.json` IS committed.

## When processing is triggered (selection-time, not pre-computed)
First click on a library thumbnail calls `POST /api/library/enhance`
(~19 s measured: source download + NLM + bilateral + diffusion). Choice
rationale: pre-computing all 300 frames ≈ 90+ min of heavy compute plus
bandwidth for images most users never open; lazy selection-time keeps
opening the grid instant and amortizes cost to viewed images only.
Tradeoff: first view of each frame waits once; every later view
(same or other user) is cache-instant. The manual-upload flow is
untouched and still processes every upload via `POST /api/enhance`.

## "Optimal settings" assumption
The library reuses the **same** `enhance_pipeline` as manual upload with
**all stages on, gamma 1.4** — the paper's full pipeline (VII.A–C: NLM
denoise → CLAHE → gamma → bilateral → anisotropic diffusion), which the
paper's results section reports as outperforming partial/traditional
variants. No separate preset exists; if presets are added later, the
cache key already includes the settings dict, so variants coexist.

## Curation (moon craters + moon dark regions only)
`backend/audit_library.py` checks every frame against its NASA record
(title + description + keywords): keep = lunar AND (crater OR dark token).
`backend/refine_library.py` purged 124 unrelated frames (launch-clean-room
and lab photos, press events, astronaut EVA, Earthrise, spacecraft
illustrations, hardware, data maps, Europa) and topped up with crater/dark
queries (Tycho/Copernicus/terminator/crescent/eclipse/full-moon/farside/
maria/highlands/phases), each survivor HEAD-verified and re-audited.
Final manifest: **300/300 confirmed keep** (`library_audit.json` holds the
per-image verdicts).

## Sourcing (paper-derived, not hardcoded topic)
`PAPER_QUERIES` in `backend/library.py` maps paper concepts → NASA Image
API terms (polar PSRs, named craters, LCROSS, LRO, literal "permanently
shadowed region", plus crater/dark extensions: Tycho, Copernicus,
terminator, crescent, eclipse, full moon, farside, maria, highlands,
phases). `backend/build_library.py` snapshots reachable public-domain
frames (HEAD-verified thumb + `~large`/`~medium` full) into the manifest.
Re-run the builder to refresh (then re-run the audit + refine passes).

## Export
NASA asset hosts send no CORS headers, so client-side canvas export would
taint. `GET /api/library/compare?id=` composes original|enhanced +
labels + disclaimer server-side from disk cache (never reprocesses) and
the UI downloads it as one PNG.
