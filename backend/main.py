"""LUNARIS backend — OHRC/PSR enhancement pipeline.

Implements the approach of Barai, Nirmal, Negi, Kamble & Hirave (2025),
"Enhancement of Permanently Shadowed Regions (PSR) of Lunar Craters
Captured by OHRC of Chandrayaan-2" (IJRPR Vol 6(4), pp 7751-7758):

  preprocessing: NLM denoise -> CLAHE contrast -> gamma correction
  enhancement:   edge-preserving smoothing (bilateral filter)
  post:          anisotropic diffusion (Perona-Malik), edge-preserving
  evaluation:    SNR, PSNR, SSIM + FVI (contrast-visibility proxy)

Contrast work is done on the lightness channel so hue is preserved.
The pipeline stretches existing faint signal; it does NOT invent terrain.
Metrics are computed original-vs-enhanced on luminance for transparency.
"""
import base64
import hashlib
import io
import os

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from PIL import Image, ImageOps
from pydantic import BaseModel

from backend import terrain as T

from backend.library import (
    PAPER_QUERIES,
    compare_png,
    get_item,
    load_manifest,
    process_selection,
)

MAX_BYTES = 50 * 1024 * 1024  # 50 MB max
MAX_DIM = 1920  # processing cap (longest side); larger uploads are downscaled
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
# Pillow format names we actually accept (content sniffed, not MIME sniffed).
ALLOWED_PIL_FORMATS = {"JPEG", "PNG", "TIFF", "WEBP"}
# Advisory only: browsers disagree on MIME, so extension + Pillow content
# are authoritative; MIME only recovers a missing filename, never rejects.
MIME_TO_EXT = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/x-png": ".png",
    "image/tiff": ".tif",
    "image/tif": ".tif",
    "image/x-tiff": ".tif",
    "image/webp": ".webp",
    "image/x-webp": ".webp",
}
ERROR_FORMAT = (
    "Unsupported format. Please upload JPG, JPEG, PNG, TIFF, or WEBP "
    "(50 MB max)."
)
METRICS_NOTE = (
    "Demo metrics computed original-vs-enhanced on luminance. "
    "FVI here is a contrast-visibility proxy (RMS-contrast ratio) defined "
    "by this demo, not a standard formula."
)

app = FastAPI(title="LUNARIS Observatory API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://bys-exe.github.io",
        # Extra frontend origins (e.g. your Vercel domain) via env:
        # CORS_EXTRA_ORIGINS=https://psr-moon.vercel.app,https://...
        *[
            o.strip()
            for o in os.environ.get("CORS_EXTRA_ORIGINS", "").split(",")
            if o.strip()
        ],
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "service": "lunaris-observatory",
        "message": "LUNARIS API is running. Open the frontend (Vite, :5173), not this URL.",
        "health": "/api/health",
        "enhance": "POST /api/enhance (multipart: file + denoise/gamma/bilateral/diffusion)",
        "analyze": "POST /api/analyze (multipart: enhanced image + classical CV params)",
        "route": "POST /api/route (JSON: analysis_id + start/target + mode) — A*/Dijkstra, no AI",
    }


@app.get("/api/health")
def health():
    return {"status": "operational", "service": "lunaris-observatory"}


def _ext_of(filename: str) -> str:
    name = (filename or "").lower()
    dot = name.rfind(".")
    return name[dot:] if dot != -1 else ""


def _as_bool(value: str, default: bool = True) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _as_gamma(value: str, default: float = 1.4) -> float:
    try:
        g = float(value)
    except (TypeError, ValueError):
        return default
    return min(3.0, max(0.5, g))


def _as_float(value: str, default: float, lo: float, hi: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return min(hi, max(lo, v))


def _as_int(value: str, default: int, lo: int, hi: int) -> int:
    try:
        v = int(float(value))
    except (TypeError, ValueError):
        return default
    return min(hi, max(lo, v))


def _gamma_lut(gamma: float) -> np.ndarray:
    inv = 1.0 / gamma
    lut = ((np.arange(256) / 255.0) ** inv * 255.0).clip(0, 255)
    return lut.astype(np.uint8)


def _anisodiff(
    gray: np.ndarray, n_iter: int = 8, kappa: float = 20.0, lam: float = 0.15
) -> np.ndarray:
    """Perona-Malik anisotropic diffusion (exponential conduction).

    Smooths flat noisy areas while preserving edges. Implemented on
    luminance only.
    """
    img = gray.astype(np.float32)
    for _ in range(n_iter):
        d_n = np.roll(img, -1, axis=0) - img
        d_s = np.roll(img, 1, axis=0) - img
        d_e = np.roll(img, -1, axis=1) - img
        d_w = np.roll(img, 1, axis=1) - img
        c_n = np.exp(-((d_n / kappa) ** 2))
        c_s = np.exp(-((d_s / kappa) ** 2))
        c_e = np.exp(-((d_e / kappa) ** 2))
        c_w = np.exp(-((d_w / kappa) ** 2))
        img += lam * (c_n * d_n + c_s * d_s + c_e * d_e + c_w * d_w)
    return np.clip(img, 0, 255).astype(np.uint8)


def _snr_db(gray: np.ndarray) -> float:
    mean = float(np.mean(gray))
    std = float(np.std(gray)) + 1e-9
    return float(20.0 * np.log10(mean / std))


def _psnr_db(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    x = a.astype(np.float64)
    y = b.astype(np.float64)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
    mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
    sig_x2 = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x * mu_x
    sig_y2 = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y * mu_y
    sig_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_x * mu_y
    num = (2 * mu_x * mu_y + c1) * (2 * sig_xy + c2)
    den = (mu_x**2 + mu_y**2 + c1) * (sig_x2 + sig_y2 + c2)
    return float(np.mean(num / (den + 1e-12)))


def enhance_pipeline(
    img: Image.Image,
    do_denoise: bool,
    gamma_v: float,
    do_bilateral: bool,
    do_diffusion: bool,
):
    """Paper pipeline. Returns (PIL RGB, stages, orig_gray, enh_gray)."""
    stages = ["normalize"]
    img = ImageOps.exif_transpose(img)

    if img.mode in ("RGBA", "LA", "PA"):
        bg = Image.new("RGB", img.size, (0, 0, 0))
        bg.paste(img.convert("RGB"), mask=img.split()[-1])
        img = bg
    elif img.mode == "P":
        img = img.convert("RGB")
    elif img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    if img.mode == "L":
        img = img.convert("RGB")

    downscaled = False
    if max(img.size) > MAX_DIM:
        img.thumbnail((MAX_DIM, MAX_DIM), Image.LANCZOS)
        downscaled = True
        stages.append(f"downscale-{MAX_DIM}px")

    rgb = np.array(img.convert("RGB"))
    orig_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # 1. Preprocessing: NLM denoise (paper VII.A).
    if do_denoise:
        rgb = cv2.fastNlMeansDenoisingColored(
            rgb, None, h=8, hColor=8,
            templateWindowSize=7, searchWindowSize=21,
        )
        stages.append("nlm-denoise")

    # 2-3. Preprocessing: CLAHE + gamma on lightness only (paper VII.A).
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lch, ach, bch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    lch = clahe.apply(lch)
    stages.append("clahe")
    if abs(gamma_v - 1.0) > 1e-9:
        lch = cv2.LUT(lch, _gamma_lut(gamma_v))
        stages.append(f"gamma-{gamma_v:g}")
    rgb = cv2.cvtColor(cv2.merge((lch, ach, bch)), cv2.COLOR_LAB2RGB)

    # 4. Enhancement: edge-preserving smoothing (paper VII.B).
    if do_bilateral:
        rgb = cv2.bilateralFilter(rgb, d=5, sigmaColor=75, sigmaSpace=75)
        stages.append("bilateral")

    # 5. Post: anisotropic diffusion on luminance (paper VII.C).
    if do_diffusion:
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        lch, ach, bch = cv2.split(lab)
        lch = _anisodiff(lch)
        rgb = cv2.cvtColor(cv2.merge((lch, ach, bch)), cv2.COLOR_LAB2RGB)
        stages.append("anisotropic-diffusion")

    enh_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return Image.fromarray(rgb), stages, orig_gray, enh_gray, downscaled


@app.post("/api/enhance")
async def enhance(
    file: UploadFile = File(...),
    denoise: str = Form("true"),
    gamma: str = Form("1.4"),
    bilateral: str = Form("true"),
    diffusion: str = Form("true"),
):
    # Extension is authoritative; MIME only recovers a missing filename.
    ext = _ext_of(file.filename)
    if not ext:
        ext = MIME_TO_EXT.get((file.content_type or "").lower(), "")
    if ext not in ALLOWED_EXTS:
        return JSONResponse(status_code=400, content={"detail": ERROR_FORMAT})

    raw = await file.read()
    if not raw:
        return JSONResponse(
            status_code=400,
            content={"detail": "Empty file. Please upload a valid lunar image."},
        )
    if len(raw) > MAX_BYTES:
        return JSONResponse(
            status_code=413,
            content={"detail": "File exceeds 50 MB limit."},
        )

    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={
                "detail": "Corrupt or unreadable image file. "
                "Please try another file."
            },
        )

    # Content-sniff: a renamed .txt/.gif must get the format error.
    if (img.format or "").upper() not in ALLOWED_PIL_FORMATS:
        return JSONResponse(status_code=400, content={"detail": ERROR_FORMAT})

    do_denoise = _as_bool(denoise, True)
    gamma_v = _as_gamma(gamma, 1.4)
    do_bilateral = _as_bool(bilateral, True)
    do_diffusion = _as_bool(diffusion, True)

    try:
        out, stages, orig_gray, enh_gray, downscaled = enhance_pipeline(
            img, do_denoise, gamma_v, do_bilateral, do_diffusion
        )
    except Exception:
        return JSONResponse(
            status_code=500,
            content={"detail": "Enhancement failed. Please try another image."},
        )

    buf = io.BytesIO()
    out.save(buf, format="PNG")
    buf.seek(0)
    image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    std_o = float(np.std(orig_gray)) + 1e-9
    std_e = float(np.std(enh_gray)) + 1e-9
    metrics = {
        "snr_before_db": round(_snr_db(orig_gray), 2),
        "snr_after_db": round(_snr_db(enh_gray), 2),
        "psnr_db": round(_psnr_db(orig_gray, enh_gray), 2),
        "ssim": round(_ssim(orig_gray, enh_gray), 4),
        "fvi_proxy": round(std_e / std_o, 3),
    }

    # Visibility check: how much more the human eye can see. Histograms
    # plus crushed-black recovery and dark-region lift, all on luminance.
    hist_before, _ = np.histogram(orig_gray, bins=256, range=(0, 256))
    hist_after, _ = np.histogram(enh_gray, bins=256, range=(0, 256))
    dark_thresh = float(np.percentile(orig_gray, 25))
    dark_mask = orig_gray <= dark_thresh
    if not bool(dark_mask.any()):
        dark_mask = np.ones_like(orig_gray, dtype=bool)
    visibility = {
        "hist_before": [int(v) for v in hist_before],
        "hist_after": [int(v) for v in hist_after],
        "crushed_before": round(float(np.mean(orig_gray < 16)) * 100.0, 1),
        "crushed_after": round(float(np.mean(enh_gray < 16)) * 100.0, 1),
        "dark_lift": round(
            (float(np.mean(enh_gray[dark_mask])) + 1.0)
            / (float(np.mean(orig_gray[dark_mask])) + 1.0),
            3,
        ),
        "contrast_gain": round(std_e / std_o, 3),
    }

    return {
        "image": "data:image/png;base64," + image_b64,
        "metrics": metrics,
        "visibility": visibility,
        "metrics_note": METRICS_NOTE,
        "stages": stages,
        "options": {
            "denoise": do_denoise,
            "gamma": gamma_v,
            "bilateral": do_bilateral,
            "diffusion": do_diffusion,
        },
        "output": {
            "width": out.size[0],
            "height": out.size[1],
            "downscaled": downscaled,
            "cap_px": MAX_DIM,
        },
    }


class LibrarySelect(BaseModel):
    id: str


@app.get("/api/library")
def library_list(q: str = "", query: str = ""):
    """Paper-sourced image grid (build-time snapshot, no live search)."""
    manifest = load_manifest()
    items = manifest.get("items", [])
    if query:
        items = [it for it in items if it.get("query") == query]
    if q:
        needle = q.strip().lower()
        items = [
            it
            for it in items
            if needle in (it.get("title") or "").lower()
            or needle in (it.get("id") or "").lower()
        ]
    return {
        "generated": manifest.get("generated"),
        "queries": [{"q": a, "rationale": b} for a, b in PAPER_QUERIES],
        "count": len(items),
        "attribution": "Public domain (NASA Image Library)",
        "items": items,
    }


@app.post("/api/library/enhance")
def library_enhance(body: LibrarySelect):
    """Selection-time enhancement with disk cache (no reprocessing)."""
    item_id = (body.id or "").strip()
    if not item_id or get_item(item_id) is None:
        return JSONResponse(
            status_code=404, content={"detail": "Library image not found."}
        )
    try:
        return process_selection(item_id)
    except ConnectionError as exc:
        return JSONResponse(status_code=502, content={"detail": str(exc)})
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"detail": str(exc)})
    except Exception:
        return JSONResponse(
            status_code=500,
            content={"detail": "Enhancement failed. Please try another image."},
        )


@app.get("/api/library/compare")
def library_compare(id: str = ""):
    """One-file side-by-side export, composed from disk cache."""
    item_id = (id or "").strip()
    if not item_id or get_item(item_id) is None:
        return JSONResponse(
            status_code=404, content={"detail": "Library image not found."}
        )
    try:
        return Response(content=compare_png(item_id), media_type="image/png")
    except (ConnectionError, ValueError) as exc:
        return JSONResponse(status_code=502, content={"detail": str(exc)})
    except Exception:
        return JSONResponse(
            status_code=500,
            content={"detail": "Enhancement failed. Please try another image."},
        )


# ---- Terrain intelligence: in-memory analysis cache ----
# key: sha256(image bytes + params). Rasters are small (analysis is capped
# at 960px); entries evicted oldest-first. Lost on restart by design —
# the frontend simply re-runs Analyze (seconds).
_ANALYSIS_CACHE: dict = {}
_ANALYSIS_ORDER: list = []
_ANALYSIS_MAX = 6
ANALYSIS_DIM = 960


def _analysis_cache_get(key: str):
    entry = _ANALYSIS_CACHE.get(key)
    if entry is not None:
        _ANALYSIS_ORDER.remove(key)
        _ANALYSIS_ORDER.append(key)
    return entry


def _analysis_cache_put(key: str, entry: dict):
    if key in _ANALYSIS_CACHE:
        _ANALYSIS_ORDER.remove(key)
    elif len(_ANALYSIS_ORDER) >= _ANALYSIS_MAX:
        _ANALYSIS_CACHE.pop(_ANALYSIS_ORDER.pop(0), None)
    _ANALYSIS_CACHE[key] = entry
    _ANALYSIS_ORDER.append(key)


def _analysis_payload(entry: dict, cached: bool) -> dict:
    a = entry["analysis"]
    return {
        "analysis_id": entry["key"],
        "cached": cached,
        "dims": a["dims"],
        "full_dims": entry["full_dims"],
        "craters": a["craters"],
        "crater_diagnostics": a["crater_diagnostics"],
        "shadow": a["shadow"],
        "hazard": a["hazard"],
        "overlays": a["overlays"],
        "params": a["params"],
        "evidence_legend": T.EVIDENCE_LEGEND,
        "no_hallucination": T.NO_HALLUCINATION,
        "note": (
            "Classical image analysis only (no AI). Depths/slopes are "
            "image-derived estimates — never true depth without elevation "
            "data. Dark pixels are candidate shadow regions, not confirmed PSRs."
        ),
    }


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    canny_lo: str = Form("50"),
    canny_hi: str = Form("150"),
    hough_acc: str = Form("30"),
    min_r: str = Form("12"),
    max_r: str = Form("200"),
    min_circ: str = Form("0.55"),
    w_slope: str = Form("0.35"),
    w_rough: str = Form("0.25"),
    w_bound: str = Form("0.25"),
    w_unc: str = Form("0.15"),
):
    """Classical terrain analysis on the (already enhanced) image."""
    ext = _ext_of(file.filename)
    if not ext:
        ext = MIME_TO_EXT.get((file.content_type or "").lower(), "")
    if ext not in ALLOWED_EXTS:
        return JSONResponse(status_code=400, content={"detail": ERROR_FORMAT})

    raw = await file.read()
    if not raw:
        return JSONResponse(
            status_code=400,
            content={"detail": "Empty file. Please upload a valid lunar image."},
        )
    if len(raw) > MAX_BYTES:
        return JSONResponse(
            status_code=413,
            content={"detail": "File exceeds 50 MB limit."},
        )

    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={
                "detail": "Corrupt or unreadable image file. "
                "Please try another file."
            },
        )
    if (img.format or "").upper() not in ALLOWED_PIL_FORMATS:
        return JSONResponse(status_code=400, content={"detail": ERROR_FORMAT})

    params = {
        "canny_lo": _as_int(canny_lo, 50, 10, 400),
        "canny_hi": _as_int(canny_hi, 150, 20, 600),
        "hough_acc": _as_int(hough_acc, 30, 10, 200),
        "min_r": _as_int(min_r, 12, 6, 400),
        "max_r": _as_int(max_r, 200, 12, 600),
        "min_circ": _as_float(min_circ, 0.55, 0.1, 1.0),
        "w_slope": _as_float(w_slope, 0.35, 0.0, 1.0),
        "w_rough": _as_float(w_rough, 0.25, 0.0, 1.0),
        "w_bound": _as_float(w_bound, 0.25, 0.0, 1.0),
        "w_unc": _as_float(w_unc, 0.15, 0.0, 1.0),
    }
    if params["min_r"] >= params["max_r"]:
        params["max_r"] = params["min_r"] + 4
    if params["canny_lo"] >= params["canny_hi"]:
        params["canny_hi"] = params["canny_lo"] + 10

    key = hashlib.sha256(
        raw + repr(sorted(params.items())).encode("utf-8")
    ).hexdigest()[:32]
    hit = _analysis_cache_get(key)
    if hit is not None:
        return _analysis_payload(hit, cached=True)

    try:
        bgr, full_dims = T.pil_to_bgr(img, ANALYSIS_DIM)
        analysis = T.analyze_image(bgr, **params)
    except Exception:
        return JSONResponse(
            status_code=500,
            content={"detail": "Terrain analysis failed. Please try another image."},
        )
    entry = {
        "key": key,
        "analysis": analysis,
        "full_dims": full_dims,
        "bgr": bgr,
    }
    _analysis_cache_put(key, entry)
    return _analysis_payload(entry, cached=False)


class RouteRequest(BaseModel):
    analysis_id: str
    start: list  # [nx, ny] normalized 0..1 in analysis dims
    target: list  # [nx, ny] normalized 0..1 in analysis dims
    mode: str = "balanced"  # shortest | safest | balanced
    algorithm: str = "astar"  # astar | dijkstra
    caution: float = 1.0


@app.post("/api/route")
def route(body: RouteRequest):
    """A*/Dijkstra rover route on cached analysis rasters."""
    entry = _analysis_cache_get((body.analysis_id or "").strip())
    if entry is None:
        return JSONResponse(
            status_code=404,
            content={"detail": "Analysis expired - please re-run Analyze terrain."},
        )
    try:
        sx, sy = float(body.start[0]), float(body.start[1])
        tx, ty = float(body.target[0]), float(body.target[1])
    except (TypeError, IndexError, ValueError):
        return JSONResponse(
            status_code=400,
            content={"detail": "Start/target must be [x, y] normalized coordinates."},
        )
    if not all(0.0 <= v <= 1.0 for v in (sx, sy, tx, ty)):
        return JSONResponse(
            status_code=400,
            content={"detail": "Coordinates must be within 0..1."},
        )
    mode = (body.mode or "balanced").lower()
    if mode not in ("shortest", "safest", "balanced"):
        mode = "balanced"
    algo = (body.algorithm or "astar").lower()
    if algo not in ("astar", "dijkstra"):
        algo = "astar"
    try:
        caution = min(3.0, max(0.0, float(body.caution)))
    except (TypeError, ValueError):
        caution = 1.0

    dims = entry["analysis"]["dims"]
    start = (sx * dims["w"], sy * dims["h"])
    target = (tx * dims["w"], ty * dims["h"])
    rasters = entry["analysis"]["_rasters"]
    try:
        result = T.plan_route(
            rasters["haz_score"], rasters["shadow_cls"], rasters["haz_class"],
            start, target, mode, caution, algo,
        )
    except Exception:
        return JSONResponse(
            status_code=500,
            content={"detail": "Route planning failed. Please try other points."},
        )
    if not result.get("found"):
        return {
            "found": False,
            "reason": result.get("reason", "No route found."),
            "pops": result.get("pops", 0),
        }
    try:
        overlay = T._b64_png(T.draw_route(entry["bgr"], result, start, target))
    except Exception:
        return JSONResponse(
            status_code=500,
            content={"detail": "Route overlay failed. Please try other points."},
        )
    result["overlay"] = overlay
    result["dims"] = dims
    result["no_hallucination"] = T.NO_HALLUCINATION
    return result
