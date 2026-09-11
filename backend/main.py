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
import io

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from PIL import Image, ImageOps
from pydantic import BaseModel

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

    return {
        "image": "data:image/png;base64," + image_b64,
        "metrics": metrics,
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
