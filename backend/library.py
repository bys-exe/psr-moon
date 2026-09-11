"""Image Library — paper-driven sourcing + cached enhancement.

What counts as a library image is derived from the research paper
(Barai et al., IJRPR 2025, "Enhancement of PSRs ... Captured by OHRC of
Chandrayaan-2"), NOT from a generic keyword:

  paper subject  -> PSR = low-light polar lunar crater floors imaged for
                    morphology / water-ice-candidate / landing-site study
  named features -> lunar south/north poles; Shackleton, Shoemaker,
                    Faustini, Cabeus craters; LCROSS (2009, Cabeus); LRO
  PAPER_QUERIES below translates each of these into a NASA Image Library
  search term, with the rationale recorded alongside the term.

Sourcing: `build_library.py` runs these queries against the public NASA
Image API (no key) and snapshots a manifest of reachable public-domain
thumbnails/full images. The app serves the snapshot, so opening the
library never depends on a live external search.

Caching: selection-time processing (see IMAGE_LIBRARY_NOTES.md for the
tradeoff). First click on a library image runs the SAME pipeline as a
manual upload (imported from main) with default/optimal settings, then
stores original + enhanced + metrics on disk keyed by content hash, so:
  - repeat views (this session) load from frontend memory: no HTTP call;
  - repeat views (other sessions/users) hit disk cache: HTTP call, zero
    reprocessing (response carries cached:true).
"""
import hashlib
import io
import json
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
MANIFEST_PATH = HERE / "library_manifest.json"
CACHE_ROOT = HERE / ".cache" / "library"

# Bump when the pipeline or defaults change -> old cache entries stop matching.
PIPELINE_VERSION = "v1"
DEFAULT_SETTINGS = {
    "denoise": True,
    "gamma": 1.4,
    "bilateral": True,
    "diffusion": True,
}

# (query, why — traceable to the paper)
# First 8 seeded the snapshot from paper concepts; the rest extend the
# paper's crater/dark definition per the moon-only curation rule
# (moon craters + moon dark regions; see refine_library.py + notes).
PAPER_QUERIES = [
    ("lunar south pole", "PSRs lie near the poles; paper studies polar craters"),
    ("Shackleton crater", "named south-pole crater in PSR literature"),
    ("Cabeus crater", "LCROSS 2009 impact site that confirmed water in a PSR"),
    ("lunar permanently shadowed region", "literal subject of the paper"),
    ("moon crater shadow", "shadowed crater floors = the low-light input regime"),
    ("lunar reconnaissance orbiter", "LRO/LOLA/Diviner context instruments"),
    ("LCROSS moon", "water-confirmation mission cited by the paper"),
    ("lunar north pole", "north-pole PSRs mirror the south-pole case"),
    ("lunar crater", "core: crater imagery for morphology study"),
    ("Tycho crater moon", "iconic rayed crater, mare backdrop"),
    ("Copernicus crater moon", "iconic terraced crater"),
    ("Clavius crater moon", "large southern crater floor"),
    ("moon terminator", "day/night boundary = dark-region imagery"),
    ("crescent moon", "mostly-dark disk phase"),
    ("lunar eclipse", "dark-phase moon"),
    ("full moon", "maria + crater detail (dark plains)"),
    ("lunar farside", "far-side cratered terrain"),
    ("lunar maria", "dark plains = moon dark regions"),
    ("lunar highlands", "heavily cratered terrain"),
    ("moon phases", "phase series showing dark portions"),
]

NASA_SEARCH = "https://images-api.nasa.gov/search"
UA = {"User-Agent": "lunaris-observatory-demo/1.0"}
FETCH_TIMEOUT = 30
FETCH_CAP = 25 * 1024 * 1024  # source-image download cap (processing caps at 1920px anyway)
ALLOWED_PIL_FORMATS = {"JPEG", "PNG", "TIFF", "WEBP"}

_manifest_cache = None


def load_manifest() -> dict:
    """Serve the build-time snapshot (empty items if never built)."""
    global _manifest_cache
    if _manifest_cache is None:
        try:
            _manifest_cache = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _manifest_cache = {"generated": None, "queries": [], "items": []}
    return _manifest_cache


def get_item(item_id: str) -> dict | None:
    for it in load_manifest().get("items", []):
        if it.get("id") == item_id:
            return it
    return None


def cache_key(item_id: str, settings: dict | None = None) -> str:
    s = settings or DEFAULT_SETTINGS
    payload = "|".join(
        [PIPELINE_VERSION, item_id, str(s)]
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:32]


def _key_dir(key: str) -> Path:
    d = CACHE_ROOT / key
    d.mkdir(parents=True, exist_ok=True)
    return d


def read_cache(key: str) -> dict | None:
    d = CACHE_ROOT / key
    meta = d / "meta.json"
    enh = d / "enhanced.png"
    orig = d / "original"
    if not (meta.exists() and enh.exists() and orig.exists()):
        return None
    try:
        return json.loads(meta.read_text(encoding="utf-8"))
    except ValueError:
        return None


def fetch_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT) as r:
        chunks, total = [], 0
        while True:
            b = r.read(65536)
            if not b:
                break
            total += len(b)
            if total > FETCH_CAP:
                raise ValueError("Source image exceeds size cap.")
            chunks.append(b)
    return b"".join(chunks)


def process_selection(item_id: str) -> dict:
    """Enhance one library image with default settings, disk-cached.

    Returns the same payload shape as POST /api/enhance, plus
    {id, title, cached}. Raises LookupError (unknown id) or
    ConnectionError/ValueError (source fetch or image problems).
    """
    from backend.main import (  # deferred: main imports this module for routes
        _psnr_db,
        _snr_db,
        _ssim,
        enhance_pipeline,
    )

    item = get_item(item_id)
    if item is None:
        raise LookupError(f"Unknown library image: {item_id!r}")

    key = cache_key(item_id)
    hit = read_cache(key)
    if hit is not None:
        payload = dict(hit["payload"])
        payload["cached"] = True
        return payload

    try:
        raw = fetch_bytes(item["full"])
    except Exception as exc:
        raise ConnectionError(
            "Original image unavailable (source offline). Try another."
        ) from exc

    import numpy as np
    from PIL import Image
    import cv2

    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception as exc:
        raise ValueError(
            "Corrupt or unreadable image file. Please try another file."
        ) from exc
    if (img.format or "").upper() not in ALLOWED_PIL_FORMATS:
        raise ValueError(
            "Unsupported format. Please upload JPG, JPEG, PNG, TIFF, or WEBP "
            "(50 MB max)."
        )

    s = DEFAULT_SETTINGS
    out, stages, orig_gray, enh_gray, downscaled = enhance_pipeline(
        img, s["denoise"], s["gamma"], s["bilateral"], s["diffusion"]
    )

    buf = io.BytesIO()
    out.save(buf, format="PNG")
    png = buf.getvalue()

    import base64

    std_o = float(np.std(orig_gray)) + 1e-9
    std_e = float(np.std(enh_gray)) + 1e-9
    from backend.main import METRICS_NOTE

    payload = {
        "id": item_id,
        "title": item.get("title", item_id),
        "image": "data:image/png;base64," + base64.b64encode(png).decode("ascii"),
        "metrics": {
            "snr_before_db": round(_snr_db(orig_gray), 2),
            "snr_after_db": round(_snr_db(enh_gray), 2),
            "psnr_db": round(_psnr_db(orig_gray, enh_gray), 2),
            "ssim": round(_ssim(orig_gray, enh_gray), 4),
            "fvi_proxy": round(std_e / std_o, 3),
        },
        "metrics_note": METRICS_NOTE,
        "stages": stages,
        "options": {"denoise": True, "gamma": 1.4, "bilateral": True, "diffusion": True},
        "output": {
            "width": out.size[0],
            "height": out.size[1],
            "downscaled": downscaled,
            "cap_px": 1920,
        },
        "cached": False,
    }

    d = _key_dir(key)
    (d / "original").write_bytes(raw)
    (d / "enhanced.png").write_bytes(png)
    (d / "meta.json").write_text(
        json.dumps(
            {"id": item_id, "key": key, "payload": payload},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return payload


def compare_png(item_id: str) -> bytes:
    """Side-by-side original|enhanced PNG for one-click export.

    Served from disk cache (processes once via process_selection if needed),
    so export never re-runs the pipeline.
    """
    from PIL import Image, ImageDraw

    payload = process_selection(item_id)  # cached after first run
    key = cache_key(item_id)
    d = CACHE_ROOT / key
    cmp_path = d / "compare.png"
    if cmp_path.exists():
        return cmp_path.read_bytes()

    orig = Image.open(d / "original").convert("RGB")
    enh = Image.open(d / "enhanced.png").convert("RGB")
    h = 720

    def fit(im):
        w = max(1, round(im.size[0] * h / im.size[1]))
        return im.resize((w, h), Image.LANCZOS)

    left, right = fit(orig), fit(enh)
    bar, gap, cap = 34, 8, 44
    W = left.size[0] + right.size[0] + gap
    H = bar + h + cap
    canvas = Image.new("RGB", (W, H), (5, 7, 13))
    dr = ImageDraw.Draw(canvas)
    dr.text((12, 9), "ORIGINAL", fill=(125, 211, 252))
    dr.text((left.size[0] + gap + 12, 9), "ENHANCED", fill=(125, 211, 252))
    canvas.paste(left, (0, bar))
    canvas.paste(right, (left.size[0] + gap, bar))
    caption = "Enhanced for visibility — brightness/contrast stretched. Not a scientific measurement of ice."
    dr.text((12, bar + h + 12), caption[:110], fill=(143, 160, 181))
    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    data = buf.getvalue()
    cmp_path.write_bytes(data)
    return data
