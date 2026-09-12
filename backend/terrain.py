"""LUNARIS terrain intelligence — classical computer vision only.

NO AI / ML / neural networks / generative models. Everything here is
OpenCV + NumPy + heapq (stdlib). Deterministic: same input + same params
-> byte-identical outputs (no RNG anywhere in this module).

Evidence classes (every output is tagged with one):
  DIRECTLY OBSERVED   — visible in the source imagery
  CLASSICALLY DERIVED — computed from observed pixels (edges, geometry)
  ESTIMATED           — math that leans on assumptions (no DEM => apparent only)
  UNKNOWN             — cannot be determined from the available data

No-hallucination principle: enhanced imagery improves interpretation of
visible structures. It does NOT reconstruct physically unobservable
terrain. There is no DEM/elevation input, so every depth/slope number is
an IMAGE-DERIVED ESTIMATE, never true depth.
"""

import base64
import heapq
import io

import cv2
import numpy as np
from PIL import Image

EVIDENCE_OBSERVED = "DIRECTLY OBSERVED"
EVIDENCE_DERIVED = "CLASSICALLY DERIVED"
EVIDENCE_ESTIMATED = "ESTIMATED (image-derived, no elevation data)"
EVIDENCE_UNKNOWN = "UNKNOWN"

NO_HALLUCINATION = (
    "Enhanced imagery improves interpretation of visible structures. "
    "It does not reconstruct physically unobservable terrain."
)

EVIDENCE_LEGEND = [
    {"tag": EVIDENCE_OBSERVED, "meaning": "Feature directly visible in source imagery."},
    {"tag": EVIDENCE_DERIVED, "meaning": "Feature calculated from observed pixels (edges, contours, geometry)."},
    {"tag": EVIDENCE_ESTIMATED, "meaning": "Math estimate leaning on assumptions; apparent depth/slope only, never true depth without elevation data."},
    {"tag": EVIDENCE_UNKNOWN, "meaning": "Cannot be determined from the available data (e.g. true PSR permanence, hidden terrain)."},
]

# Shadow-class raster codes (Module 2).
CLS_ILLUMINATED = 0
CLS_TRANSITION = 1
CLS_SHADOW = 2

# Hazard-class raster codes (Module 3).
HAZ_LOW = 0
HAZ_MODERATE = 1
HAZ_HIGH = 2
HAZ_UNKNOWN = 3

HAZARD_NAMES = ["LOW RISK", "MODERATE RISK", "HIGH RISK", "UNKNOWN"]

# BGR overlay colours.
COL_CRATER = (255, 220, 90)      # pale yellow rims (fitted circle)
COL_TRACED = (220, 80, 255)      # magenta trace of the real uneven rim
COL_DIAM = (240, 240, 240)       # white diameter chord
COL_RADIUS = (110, 255, 150)     # green radius spoke
COL_TEXT = (255, 255, 255)       # white measurement labels
COL_SHADOW_EDGE = (0, 140, 255)  # orange PSR-candidate boundary
COL_ILLUM = (0, 170, 0)          # green
COL_TRANS = (0, 215, 255)        # yellow
COL_SHADOW = (30, 30, 30)        # near-black
COL_HAZ = [(0, 180, 0), (0, 215, 255), (0, 0, 255), (150, 150, 150)]
COL_START = (255, 80, 40)        # blue marker
COL_TARGET = (60, 60, 230)       # red destination marker
COL_ROUTE = (120, 255, 255)      # bright route line


def _b64_png(bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:  # pragma: no cover - encoder failure is near-impossible
        raise ValueError("Overlay PNG encoding failed.")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _norm01(f: np.ndarray) -> np.ndarray:
    f = f.astype(np.float64)
    lo, hi = float(np.percentile(f, 1.0)), float(np.percentile(f, 99.0))
    if hi - lo < 1e-9:
        return np.zeros_like(f)
    return np.clip((f - lo) / (hi - lo), 0.0, 1.0)


def _simplify_poly(pts: np.ndarray, max_pts: int = 180) -> np.ndarray:
    """approxPolyDP simplification with adaptive epsilon. Deterministic."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 3:
        return pts
    eps = 1.5
    approx = pts
    for _ in range(6):
        approx = cv2.approxPolyDP(pts.astype(np.float32), eps, True).reshape(-1, 2)
        if len(approx) <= max_pts or eps > 32.0:
            break
        eps *= 1.6
    return np.asarray(approx, dtype=np.float64)


def _circle_poly(cx: float, cy: float, r: float, n: int = 72) -> np.ndarray:
    """Nominal circle as a polyline (last-resort fallback only)."""
    a = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a)])


def _valid_carried_poly(poly, cx: float, cy: float, r: float) -> np.ndarray | None:
    """Accept a detector contour as the rim trace only if it agrees with the
    merged center/radius. Otherwise it belongs to a different structure."""
    try:
        p = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    except (ValueError, TypeError):
        return None
    if len(p) < 8:
        return None
    m = cv2.moments(p.astype(np.float32))
    if m["m00"] < 1e-9:
        return None
    mx, my = float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"])
    if np.hypot(mx - cx, my - cy) > 0.6 * r:
        return None
    mean_r = float(np.mean(np.hypot(p[:, 0] - mx, p[:, 1] - my)))
    if not (0.5 * r <= mean_r <= 1.6 * r):
        return None
    return _simplify_poly(p)


def _trace_radial(
    edges: np.ndarray,
    grad: np.ndarray,
    cx: float,
    cy: float,
    r_nom: float,
    w: int,
    h: int,
    n_angles: int = 144,
) -> np.ndarray | None:
    """Trace the observed rim by radial profile: at each angle keep the edge
    pixel nearest the nominal radius. Follows uneven rims because every
    vertex is an observed edge pixel — never a fitted circle."""
    r_lo = max(4.0, 0.55 * r_nom)
    r_hi = 1.45 * r_nom
    pts = []
    for k in range(n_angles):
        a = 2.0 * np.pi * k / n_angles
        dx, dy = float(np.cos(a)), float(np.sin(a))
        best = None
        rad = r_lo
        while rad <= r_hi:
            px = int(round(cx + rad * dx))
            py = int(round(cy + rad * dy))
            if 0 <= px < w and 0 <= py < h and edges[py, px]:
                d = abs(rad - r_nom)
                g = float(grad[py, px])
                if best is None or (d, -g) < (best[0], best[1]):
                    best = (d, -g, float(px), float(py))
            rad += 1.0
        if best is not None:
            pts.append([best[2], best[3]])
    if len(pts) < int(0.4 * n_angles):
        return None
    return _simplify_poly(np.array(pts, dtype=np.float64))


def _feret_diameter(poly: np.ndarray) -> tuple:
    """Longest rim-to-rim distance (Feret max) + its endpoints. The diameter
    chord is drawn along this axis, so uneven craters are measured along
    their own long axis instead of a fixed horizontal line."""
    hull = cv2.convexHull(poly.astype(np.float32)).reshape(-1, 2).astype(np.float64)
    best = (0.0, hull[0], hull[0] if len(hull) == 1 else hull[-1])
    for i in range(len(hull)):
        for j in range(i + 1, len(hull)):
            d = float(np.hypot(hull[j, 0] - hull[i, 0], hull[j, 1] - hull[i, 1]))
            if d > best[0]:
                best = (d, hull[i], hull[j])
    return best  # (length, p1, p2)


def _ray_hit(px: float, py: float, dx: float, dy: float,
             poly: np.ndarray) -> tuple | None:
    """Nearest intersection of ray (p + t·d, t > 0) with a closed polyline."""
    best = None
    n = len(poly)
    for i in range(n):
        x1, y1 = float(poly[i, 0]), float(poly[i, 1])
        x2, y2 = float(poly[(i + 1) % n, 0]), float(poly[(i + 1) % n, 1])
        ex, ey = x2 - x1, y2 - y1
        denom = dx * ey - dy * ex
        if abs(denom) < 1e-12:
            continue
        qx, qy = x1 - px, y1 - py
        t = (qx * ey - qy * ex) / denom
        s = (qx * dy - qy * dx) / denom
        if t > 0 and 0.0 <= s <= 1.0 and (best is None or t < best[0]):
            best = (t, px + t * dx, py + t * dy)
    return None if best is None else (best[1], best[2])


def gradient_maps(gray: np.ndarray) -> dict:
    """Sobel gradient magnitude + local-std roughness. CLASSICALLY DERIVED."""
    g = gray.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    mean = cv2.boxFilter(g, -1, (15, 15))
    sq = cv2.boxFilter(g * g, -1, (15, 15))
    rough = np.sqrt(np.maximum(sq - mean * mean, 0.0))
    return {"grad": grad, "rough": rough}


def shadow_segmentation(gray: np.ndarray) -> dict:
    """Otsu + adaptive threshold fusion into illuminated/transition/shadow.

    A dark pixel is NOT automatically a PSR: classes are OBSERVABLE shadow
    regions; permanence is UNKNOWN without illumination data.
    """
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    t_otsu, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t_otsu = float(t_otsu)
    shadow = (blur < 0.80 * t_otsu).astype(np.uint8) * 255
    trans = ((blur >= 0.80 * t_otsu) & (blur < 1.25 * t_otsu)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    shadow = cv2.morphologyEx(shadow, cv2.MORPH_OPEN, kernel)
    shadow = cv2.morphologyEx(
        shadow, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    )
    cls = np.full(gray.shape, CLS_ILLUMINATED, dtype=np.uint8)
    cls[trans.astype(bool)] = CLS_TRANSITION
    cls[shadow.astype(bool)] = CLS_SHADOW

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(shadow, 8)
    min_area = max(25.0, gray.size * 0.0002)
    regions = []
    contours, _ = cv2.findContours(shadow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perim_by_centroid = {}
    for c in contours:
        m = cv2.moments(c)
        if m["m00"] > 0:
            perim_by_centroid[(round(m["m10"] / m["m00"]), round(m["m01"] / m["m00"]))] = float(
                cv2.arcLength(c, True)
            )
    for i in range(1, n):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        cx, cy = float(centroids[i][0]), float(centroids[i][1])
        perim = perim_by_centroid.get((round(cx), round(cy)), 0.0)
        if perim <= 0:
            # Fall back to bounding-box perimeter when contour matching misses.
            w = float(stats[i, cv2.CC_STAT_WIDTH])
            h = float(stats[i, cv2.CC_STAT_HEIGHT])
            perim = 2.0 * (w + h)
        compact = (4.0 * np.pi * area / (perim * perim)) if perim > 0 else 0.0
        regions.append({
            "id": len(regions),
            "centroid": [round(cx, 1), round(cy, 1)],
            "area_px": int(area),
            "area_pct": round(100.0 * area / gray.size, 3),
            "perimeter_px": round(perim, 1),
            "bbox": [int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                     int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])],
            "compactness": round(min(compact, 1.0), 3),
            "class": "CANDIDATE PSR REGION",
            "class_evidence": EVIDENCE_DERIVED,
            "permanence": EVIDENCE_UNKNOWN,
        })
    regions.sort(key=lambda r: r["area_px"], reverse=True)
    for k, r in enumerate(regions):
        r["id"] = k
    return {
        "class_raster": cls,
        "otsu_threshold": round(t_otsu, 1),
        "shadow_pct": round(100.0 * float(np.mean(cls == CLS_SHADOW)), 2),
        "transition_pct": round(100.0 * float(np.mean(cls == CLS_TRANSITION)), 2),
        "regions": regions,
    }


def find_craters(
    gray: np.ndarray,
    grad: np.ndarray,
    canny_lo: int = 50,
    canny_hi: int = 150,
    hough_acc: int = 30,
    min_r: int = 12,
    max_r: int = 200,
    min_circularity: float = 0.55,
    max_craters: int = 40,
) -> dict:
    """Two-detector crater search: Hough circles x contour circularity.

    Neither detector is trusted alone; the evidence score rewards agreement
    plus rim continuity sampled directly on the Canny edge map.
    """
    min_r = max(6, int(min_r))
    max_r = max(min_r + 4, int(max_r))
    blur = cv2.medianBlur(gray, 5)
    edges = cv2.Canny(blur, int(canny_lo), int(canny_hi))
    edge_dil = cv2.dilate(edges, np.ones((3, 3), np.uint8))

    hough = []
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, 1.2,
        minDist=max(20, min_r * 2),
        param1=int(canny_hi), param2=int(hough_acc),
        minRadius=min_r, maxRadius=max_r,
    )
    if circles is not None:
        for x, y, r in circles[0]:
            hough.append((float(x), float(y), float(r)))

    conts = []
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Raw edge contours, kept for uneven-rim tracing below. Unlike `conts`
    # (which only admits near-circular loops), these keep every sizable
    # contour so jagged / partial crater walls can still be traced.
    edge_contours = []
    for c in contours:
        _area = float(cv2.contourArea(c))
        _perim = float(cv2.arcLength(c, True))
        if _perim > 12.0 and _area > 20.0:
            _m = cv2.moments(c)
            if _m["m00"] > 1e-9:
                edge_contours.append({
                    "pts": c,
                    "area": _area,
                    "perim": _perim,
                    "cx": float(_m["m10"] / _m["m00"]),
                    "cy": float(_m["m01"] / _m["m00"]),
                })
    for c in contours:
        area = float(cv2.contourArea(c))
        perim = float(cv2.arcLength(c, True))
        if perim < 1e-9 or not (np.pi * min_r**2 * 0.5 < area < np.pi * max_r**2 * 1.5):
            continue
        circ = 4.0 * np.pi * area / (perim * perim)
        if circ < float(min_circularity):
            continue
        m = cv2.moments(c)
        if m["m00"] < 1e-9:
            continue
        cx, cy = float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"])
        r = float(np.sqrt(area / np.pi))
        if not (min_r <= r <= max_r):
            continue
        ecc = 0.0
        if len(c) >= 5:
            (_, _), (w, h), _ = cv2.fitEllipse(c)
            a, b = max(w, h) / 2.0, min(w, h) / 2.0
            ecc = float(np.sqrt(max(0.0, 1.0 - (b / a) ** 2))) if a > 0 else 0.0
        conts.append({"x": cx, "y": cy, "r": r, "circ": min(circ, 1.0), "ecc": ecc})

    h, w = gray.shape
    rim_support = edge_dil.astype(bool)

    def rim_completeness(x: float, y: float, r: float, samples: int = 72) -> float:
        hits = 0
        for k in range(samples):
            a = 2.0 * np.pi * k / samples
            px = int(round(x + r * np.cos(a)))
            py = int(round(y + r * np.sin(a)))
            if 0 <= px < w and 0 <= py < h and rim_support[py, px]:
                hits += 1
        return hits / samples

    pool = (
        [{"x": x, "y": y, "r": r, "src": "hough"} for x, y, r in hough]
        + [{**c, "src": "contour"} for c in conts]
    )
    # Greedy non-maximum suppression: strongest source first.
    order = {"both": 0, "contour": 1, "hough": 2}
    pool.sort(key=lambda c: order.get(c["src"], 3))
    kept = []
    for c in pool:
        if any(
            np.hypot(c["x"] - k["x"], c["y"] - k["y"]) < 0.6 * max(c["r"], k["r"])
            for k in kept
        ):
            # Merge: mark agreement instead of duplicating.
            for k in kept:
                if np.hypot(c["x"] - k["x"], c["y"] - k["y"]) < 0.6 * max(c["r"], k["r"]):
                    if {c["src"], k["src"]} >= {"hough", "contour"} or (
                        c["src"] != k["src"]
                    ):
                        k["src"] = "both"
                        k["x"] = (k["x"] + c["x"]) / 2.0
                        k["y"] = (k["y"] + c["y"]) / 2.0
                        k["r"] = (k["r"] + c["r"]) / 2.0
                        if "circ" not in k and "circ" in c:
                            k["circ"] = c["circ"]
                        if "ecc" not in k and "ecc" in c:
                            k["ecc"] = c["ecc"]
                    break
            continue
        kept.append(dict(c))

    craters = []
    yy, xx = np.mgrid[0:h, 0:w]
    for c in kept[: max_craters * 2]:
        x, y, r = c["x"], c["y"], c["r"]
        rim = rim_completeness(x, y, r)
        dist = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
        band = (np.abs(dist - r) < max(2.0, r * 0.12)) & (dist < w) & (dist < h)
        inner = dist < r * 0.7
        rim_g = float(np.mean(grad[band])) if bool(band.any()) else 0.0
        floor = float(np.mean(gray[inner])) if bool(inner.any()) else float(np.mean(gray))
        rim_b = float(np.mean(gray[band])) if bool(band.any()) else floor
        # Apparent depth ONLY: rim-to-floor brightness relief, divided by
        # full scale. Illumination-dependent; never true depth.
        depth_indicator = float(np.clip((rim_b - floor) / 255.0, 0.0, 1.0))
        base = 0.50 if c["src"] == "hough" else (0.65 if c["src"] == "contour" else 0.90)
        evidence = min(0.95, base + (0.05 if rim > 0.6 else 0.0))
        craters.append({
            "center": [round(x, 1), round(y, 1)],
            "radius_px": round(r, 1),
            "diameter_px": round(2.0 * r, 1),
            "circumference_px": round(2.0 * np.pi * r, 1),
            "circularity": round(float(c.get("circ", 0.0)), 3),
            "eccentricity": round(float(c.get("ecc", 0.0)), 3),
            "rim_completeness": round(rim, 3),
            "local_gradient": round(rim_g, 1),
            "depth_indicator": round(depth_indicator, 3),
            "depth_label": "Apparent depth (image-derived; NOT true depth)",
            "detectors": c["src"],
            "evidence": round(evidence, 2),
            "evidence_class": EVIDENCE_DERIVED,
        })
    craters.sort(key=lambda c: (-c["evidence"], -c["diameter_px"]))
    craters = craters[:max_craters]
    for k, c in enumerate(craters):
        c["id"] = f"C{k + 1:02d}"

    # Rim mask for hazard proximity + overlay.
    rim_mask = np.zeros((h, w), dtype=np.uint8)
    for c in craters:
        cv2.circle(rim_mask, (int(round(c["center"][0])), int(round(c["center"][1]))),
                   int(round(c["radius_px"])), 255, 3)
    return {
        "craters": craters,
        "rim_mask": rim_mask,
        "hough_count": len(hough),
        "contour_count": len(conts),
        "edge_px": int(np.count_nonzero(edges)),
    }


def hazard_analysis(
    gray: np.ndarray,
    grad: np.ndarray,
    rough: np.ndarray,
    rim_mask: np.ndarray,
    shadow_cls: np.ndarray,
    w_slope: float = 0.35,
    w_rough: float = 0.25,
    w_bound: float = 0.25,
    w_unc: float = 0.15,
) -> dict:
    """Transparent weighted hazard raster. Weights are user-configurable."""
    ws = np.array([w_slope, w_rough, w_bound, w_unc], dtype=np.float64)
    ws = np.clip(ws, 0.0, 1.0)
    if ws.sum() < 1e-9:
        ws = np.array([0.35, 0.25, 0.25, 0.15])
    ws = ws / ws.sum()

    slope = _norm01(grad)
    rough_n = _norm01(rough)
    inv_rim = (255 - rim_mask).astype(np.uint8)
    dist = cv2.distanceTransform(inv_rim, cv2.DIST_L2, 3)
    prox = 1.0 - np.clip(dist / (0.10 * max(gray.shape)), 0.0, 1.0)
    unc = np.where(shadow_cls == CLS_SHADOW, 0.8,
                   np.where(shadow_cls == CLS_TRANSITION, 0.4, 0.0))
    score = ws[0] * slope + ws[1] * rough_n + ws[2] * prox + ws[3] * unc

    hcls = np.full(gray.shape, HAZ_LOW, dtype=np.uint8)
    hcls[score >= 0.33] = HAZ_MODERATE
    hcls[score >= 0.66] = HAZ_HIGH
    # Unknown: shadowed AND nearly featureless => no information to judge.
    grad_p5 = float(np.percentile(grad, 5))
    unknown = (shadow_cls == CLS_SHADOW) & (grad <= grad_p5)
    hcls[unknown] = HAZ_UNKNOWN

    counts = {HAZARD_NAMES[k]: round(100.0 * float(np.mean(hcls == k)), 2) for k in range(4)}
    contrib = {
        "slope": round(float(ws[0] * np.mean(slope)), 4),
        "roughness": round(float(ws[1] * np.mean(rough_n)), 4),
        "crater_boundary": round(float(ws[2] * np.mean(prox)), 4),
        "uncertainty": round(float(ws[3] * np.mean(unc)), 4),
    }
    return {
        "score": score.astype(np.float32),
        "class_raster": hcls,
        "weights": {"slope": round(float(ws[0]), 3), "roughness": round(float(ws[1]), 3),
                    "boundary": round(float(ws[2]), 3), "uncertainty": round(float(ws[3]), 3)},
        "class_pct": counts,
        "mean_contrib": contrib,
        "mean_score": round(float(np.mean(score)), 4),
    }


def _label(canvas: np.ndarray, text: str, org: tuple) -> None:
    """White label with black outline so it reads over any terrain."""
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                COL_TEXT, 1, cv2.LINE_AA)


def draw_craters_clean(base_bgr: np.ndarray, craters: list) -> np.ndarray:
    """Dimmed base with measured rims: rim ring + diameter chord with end
    ticks + radius spoke + value labels. Everything visible is measured
    on-image (CLASSICALLY DERIVED); nothing is inferred."""
    canvas = np.clip(base_bgr.astype(np.float32) * 0.45, 0, 255).astype(np.uint8)
    h, w = canvas.shape[:2]
    for c in craters:
        cx, cy = int(round(c["center"][0])), int(round(c["center"][1]))
        r = max(4, int(round(c["radius_px"])))
        # 1. Highlighted rim.
        cv2.circle(canvas, (cx, cy), r, COL_CRATER, 2)
        cv2.circle(canvas, (cx, cy), 2, COL_CRATER, -1)
        if c["detectors"] == "both":
            cv2.circle(canvas, (cx, cy), r + 4, (90, 220, 90), 1)
        # 2. Diameter chord (horizontal) with end ticks.
        y0 = min(max(cy, 0), h - 1)
        x1, x2 = cx - r, cx + r
        cv2.line(canvas, (x1, y0), (x2, y0), COL_DIAM, 1, cv2.LINE_AA)
        for xe in (x1, x2):
            cv2.line(canvas, (xe, y0 - 5), (xe, y0 + 5), COL_DIAM, 1, cv2.LINE_AA)
        # 3. Radius spoke at -35 degrees + value at its midpoint.
        ang = -35.0 * np.pi / 180.0
        ex, ey = int(round(cx + r * np.cos(ang))), int(round(cy + r * np.sin(ang)))
        cv2.line(canvas, (cx, cy), (ex, ey), COL_RADIUS, 1, cv2.LINE_AA)
        mx, my = int(round(cx + (r / 2.0) * np.cos(ang))), int(round(cy + (r / 2.0) * np.sin(ang)))
        _label(canvas, f"r{c['radius_px']}", (mx + 3, my - 3))
        # 4. Measurement tag above the rim: id + diameter + circumference.
        _label(canvas, f"{c['id']} D{c['diameter_px']}", (x1, max(cy - r - 20, 10)))
        _label(canvas, f"C{c['circumference_px']}", (x1, max(cy - r - 7, 10)))
    return canvas


def draw_shadow(base_bgr: np.ndarray, cls: np.ndarray, regions: list) -> np.ndarray:
    """Dimmed base + green/yellow/dark zones + orange candidate boundaries."""
    canvas = (base_bgr.astype(np.float32) * 0.35).copy()
    zone = np.zeros_like(base_bgr, dtype=np.float32)
    zone[cls == CLS_ILLUMINATED] = COL_ILLUM
    zone[cls == CLS_TRANSITION] = COL_TRANS
    zone[cls == CLS_SHADOW] = COL_SHADOW
    canvas = np.clip(canvas * 0.35 + zone * 0.65, 0, 255).astype(np.uint8)
    shadow_bin = (cls == CLS_SHADOW).astype(np.uint8) * 255
    contours, _ = cv2.findContours(shadow_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, COL_SHADOW_EDGE, 2)
    for r in regions[:20]:
        cx, cy = int(round(r["centroid"][0])), int(round(r["centroid"][1]))
        cv2.putText(canvas, f"PSR?{r['id']}", (cx + 6, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_SHADOW_EDGE, 1, cv2.LINE_AA)
    return canvas


def draw_hazard(base_bgr: np.ndarray, hcls: np.ndarray) -> np.ndarray:
    """Dimmed base + green/yellow/red/gray hazard zones."""
    canvas = (base_bgr.astype(np.float32) * 0.30).copy()
    zone = np.zeros_like(base_bgr, dtype=np.float32)
    for k in range(4):
        zone[hcls == k] = COL_HAZ[k]
    canvas = canvas * 0.30 + zone * 0.70
    return np.clip(canvas, 0, 255).astype(np.uint8)


def cost_grid(
    haz_score: np.ndarray,
    shadow_cls: np.ndarray,
    hazard_class: np.ndarray,
    mode: str = "balanced",
    caution: float = 1.0,
) -> np.ndarray:
    """Traversability cost raster. All weights explicit; mode = preset."""
    presets = {
        # Shortest must be distance-true: A* minimizes COST, so any large
        # penalty visibly bends the path. Tiny penalties only break ties.
        "shortest": {"haz": 0.05, "unk": 0.10, "shadow": 0.02},
        "safest": {"haz": 6.0, "unk": 8.0, "shadow": 1.5},
        "balanced": {"haz": 2.5, "unk": 3.0, "shadow": 0.6},
    }
    p = presets.get(mode, presets["balanced"])
    caution = float(np.clip(caution, 0.0, 3.0))
    cost = (
        1.0
        + p["haz"] * caution * haz_score
        + p["unk"] * caution * (hazard_class == HAZ_UNKNOWN).astype(np.float64)
        + p["shadow"] * (shadow_cls == CLS_SHADOW).astype(np.float64)
    )
    return np.clip(cost, 1.0, 50.0).astype(np.float32)


def _search(cost: np.ndarray, start: tuple, target: tuple, use_heuristic: bool):
    """A* (heuristic) or Dijkstra (no heuristic) on an 8-connected grid."""
    h, w = cost.shape
    sx, sy = int(np.clip(start[0], 0, w - 1)), int(np.clip(start[1], 0, h - 1))
    tx, ty = int(np.clip(target[0], 0, w - 1)), int(np.clip(target[1], 0, h - 1))
    if (sx, sy) == (tx, ty):
        return [(sx, sy)], float(cost[sy, sx]), 1
    dirs = [(-1, -1, 1.4142), (0, -1, 1.0), (1, -1, 1.4142),
            (-1, 0, 1.0), (1, 0, 1.0),
            (-1, 1, 1.4142), (0, 1, 1.0), (1, 1, 1.4142)]
    hmin = float(np.min(cost))
    board_cap = 1500000
    g = {(sx, sy): 0.0}
    parent = {}

    def heur(x: int, y: int) -> float:
        return hmin * np.hypot(tx - x, ty - y) if use_heuristic else 0.0

    open_heap = [(heur(sx, sy), 0.0, (sx, sy))]
    closed = set()
    pops = 0
    found = False
    while open_heap and pops < board_cap:
        _, gc, (x, y) = heapq.heappop(open_heap)
        pops += 1
        if (x, y) in closed:
            continue
        closed.add((x, y))
        if (x, y) == (tx, ty):
            found = True
            break
        for dx, dy, step in dirs:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            ng = gc + float(cost[ny, nx]) * step
            if ng < g.get((nx, ny), float("inf")):
                g[(nx, ny)] = ng
                parent[(nx, ny)] = (x, y)
                heapq.heappush(open_heap, (ng + heur(nx, ny), ng, (nx, ny)))
    if not found:
        return None, None, pops
    path = [(tx, ty)]
    while path[-1] != (sx, sy):
        path.append(parent[path[-1]])
    path.reverse()
    return path, g[(tx, ty)], pops


def smooth_path(path: list, iterations: int = 2) -> list:
    """Chaikin corner-cutting for DISPLAY ONLY.

    Grid paths are sequences of 45/90-degree steps, which render as kinked
    straight segments. Smoothing curves them. Stats (distance, hazard) are
    always computed on the true grid path, never on this curve.
    """
    pts = [(float(x), float(y)) for x, y in path]
    if len(pts) < 3:
        return pts
    for _ in range(max(0, iterations)):
        out = [pts[0]]
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
            out.append((0.75 * x0 + 0.25 * x1, 0.75 * y0 + 0.25 * y1))
            out.append((0.25 * x0 + 0.75 * x1, 0.25 * y0 + 0.75 * y1))
        out.append(pts[-1])
        pts = out
    return pts


def plan_route(
    haz_score: np.ndarray,
    shadow_cls: np.ndarray,
    hazard_class: np.ndarray,
    start: tuple,
    target: tuple,
    mode: str = "balanced",
    caution: float = 1.0,
    algorithm: str = "astar",
) -> dict:
    """Route + stats. All inputs classical rasters; output is ESTIMATED."""
    cost = cost_grid(haz_score, shadow_cls, hazard_class, mode, caution)
    path, total, pops = _search(cost, start, target, use_heuristic=(algorithm != "dijkstra"))
    if path is None:
        return {"found": False, "pops": pops,
                "reason": "No route within the search cap — try SAFEST, closer points, or lower caution."}
    step = max(1, len(path) // 800)
    slim = path[::step]
    if slim[-1] != path[-1]:
        slim.append(path[-1])
    # Display curve: Chaikin-smoothed, clipped to the raster. Stats below
    # always come from the true grid path, never from this curve.
    h, w = haz_score.shape
    disp = [
        [int(min(max(round(x), 0), w - 1)), int(min(max(round(y), 0), h - 1))]
        for x, y in smooth_path(slim)
    ]
    dist = float(sum(np.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path[:-1], path[1:])))
    cells_haz = [int(hazard_class[y, x]) for x, y in path]
    cells_score = [float(haz_score[y, x]) for x, y in path]
    unknown_n = int(sum(1 for c in cells_haz if c == HAZ_UNKNOWN))
    return {
        "found": True,
        "algorithm": "Dijkstra" if algorithm == "dijkstra" else "A*",
        "mode": mode,
        "path": [[int(x), int(y)] for x, y in slim],
        "display_path": disp,
        "path_len_cells": len(path),
        "distance_px": round(dist, 1),
        "total_cost": round(float(total), 1),
        "max_hazard": HAZARD_NAMES[max(cells_haz)],
        "avg_hazard_score": round(float(np.mean(cells_score)), 4),
        "unknown_cells": unknown_n,
        "unknown_pct": round(100.0 * unknown_n / max(len(path), 1), 2),
        "pops": pops,
        "evidence_class": EVIDENCE_ESTIMATED,
    }


def draw_route(base_bgr: np.ndarray, route: dict, start: tuple, target: tuple) -> np.ndarray:
    """Dimmed base + blue start, red target, bright path."""
    canvas = np.clip(base_bgr.astype(np.float32) * 0.55, 0, 255).astype(np.uint8)
    line = route.get("display_path") or route["path"]
    pts = np.array(line, dtype=np.int32).reshape(-1, 1, 2)
    if len(pts) >= 2:
        cv2.polylines(canvas, [pts], False, COL_ROUTE, 2, cv2.LINE_AA)
    cv2.circle(canvas, (int(start[0]), int(start[1])), 6, COL_START, -1)
    cv2.circle(canvas, (int(start[0]), int(start[1])), 9, COL_START, 2)
    cv2.drawMarker(canvas, (int(target[0]), int(target[1])), COL_TARGET,
                   cv2.MARKER_TILTED_CROSS, 18, 2, cv2.LINE_AA)
    return canvas


def analyze_image(
    bgr: np.ndarray,
    canny_lo: int = 50,
    canny_hi: int = 150,
    hough_acc: int = 30,
    min_r: int = 12,
    max_r: int = 200,
    min_circ: float = 0.55,
    w_slope: float = 0.35,
    w_rough: float = 0.25,
    w_bound: float = 0.25,
    w_unc: float = 0.15,
) -> dict:
    """Full classical chain. Returns features + overlay dataURLs + diagnostics."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    feats = gradient_maps(gray)
    shadow = shadow_segmentation(gray)
    craters = find_craters(
        gray, feats["grad"], canny_lo, canny_hi, hough_acc,
        min_r, max_r, min_circ,
    )
    haz = hazard_analysis(
        gray, feats["grad"], feats["rough"], craters["rim_mask"],
        shadow["class_raster"], w_slope, w_rough, w_bound, w_unc,
    )
    return {
        "dims": {"w": int(gray.shape[1]), "h": int(gray.shape[0])},
        "craters": craters["craters"],
        "crater_diagnostics": {
            "hough_raw": craters["hough_count"],
            "contour_raw": craters["contour_count"],
            "kept": len(craters["craters"]),
            "edge_px": craters["edge_px"],
            "agreement_rate": round(
                sum(1 for c in craters["craters"] if c["detectors"] == "both")
                / max(len(craters["craters"]), 1), 3),
        },
        "shadow": {
            "otsu_threshold": shadow["otsu_threshold"],
            "shadow_pct": shadow["shadow_pct"],
            "transition_pct": shadow["transition_pct"],
            "regions": shadow["regions"],
        },
        "hazard": {
            "weights": haz["weights"],
            "class_pct": haz["class_pct"],
            "mean_contrib": haz["mean_contrib"],
            "mean_score": haz["mean_score"],
        },
        "overlays": {
            "craters": _b64_png(draw_craters_clean(bgr, craters["craters"])),
            "shadow": _b64_png(draw_shadow(bgr, shadow["class_raster"], shadow["regions"])),
            "hazard": _b64_png(draw_hazard(bgr, haz["class_raster"])),
        },
        "_rasters": {
            "haz_score": haz["score"],
            "haz_class": haz["class_raster"],
            "shadow_cls": shadow["class_raster"],
        },
        "params": {
            "canny_lo": canny_lo, "canny_hi": canny_hi, "hough_acc": hough_acc,
            "min_r": min_r, "max_r": max_r, "min_circ": min_circ,
            "w_slope": w_slope, "w_rough": w_rough, "w_bound": w_bound, "w_unc": w_unc,
        },
    }


def pil_to_bgr(img: "Image.Image", max_dim: int = 960) -> tuple:
    """Normalise an upload to a capped BGR array. Returns (bgr, scale_info)."""
    from PIL import ImageOps

    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "PA"):
        bg = Image.new("RGB", img.size, (0, 0, 0))
        bg.paste(img.convert("RGB"), mask=img.split()[-1])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")
    full_w, full_h = img.size
    if max(full_w, full_h) > max_dim:
        img.thumbnail((max_dim, max_dim), Image.LANCZOS)
    rgb = np.array(img)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr, {"full_w": full_w, "full_h": full_h,
                 "w": bgr.shape[1], "h": bgr.shape[0]}
