"""Audit library frames against the rule: lunar AND (crater OR dark-region).

Usage (from repo root):
    python backend/audit_library.py           # audit manifest -> library_audit.json
    python backend/audit_library.py --apply   # rewrite manifest keeping verdict=keep

Verdict logic (all transparent, recorded per item):
  - text = title + description + keywords from the NASA Image API record.
  - LUNAR if moon|lunar|lro|chandrayaan|selene|grail| Clementine? (mission
    tokens alone are weak -> require moon|lunar, else 'review').
  - CRATER if crater|basin|mare|highland|rille|regolith|tycho|copernicus|kepler...
  - DARK if shadow|dark|night|terminator|eclipse|crescent|pole|polar|far side|
    earthshine|new moon.
  - keep    = LUNAR and (CRATER or DARK)
  - review  = LUNAR but neither (human adjudicates from title)
  - remove  = not LUNAR (astronauts, hardware, Earth, launches, diagrams...)
Also records mean thumbnail luminance as a supporting "dark?" signal.
"""
import concurrent.futures as cf
import io
import json
import sys
import urllib.parse
import urllib.request

sys.path.insert(0, ".")
from backend.library import MANIFEST_PATH, UA  # noqa: E402

LUNAR = ("moon", "lunar")
CRATER = (
    "crater", "basin", "mare", "maria", "highland", "rille", "regolith",
    "tycho", "copernicus", "kepler", "clavius", "plato", "aristarchus",
    "schickard", "grimaldi", "langrenus", "petavius", "theophilus",
)
DARK = (
    "shadow", "dark", "night", "terminator", "eclipse", "crescent", "pole",
    "polar", "far side", "farside", "earthshine", "new moon", "waxing",
    "waning", "gibbous",
)


def get_json(url, timeout=30):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def record(nasa_id):
    try:
        qs = urllib.parse.urlencode(
            {"nasa_id": nasa_id, "media_type": "image"}
        )
        data = get_json(f"https://images-api.nasa.gov/search?{qs}")
        items = data.get("collection", {}).get("items", [])
        d = items[0]["data"][0] if items else {}
    except Exception as exc:
        return {"id": nasa_id, "error": str(exc)[:120]}
    return {
        "id": nasa_id,
        "title": d.get("title", ""),
        "description": (d.get("description") or "")[:600],
        "keywords": d.get("keywords") or [],
        "center": d.get("center", ""),
    }


def brightness(thumb_url):
    """Mean 0-255 luminance of the thumbnail (supporting signal only)."""
    try:
        req = urllib.request.Request(thumb_url, headers=UA)
        with urllib.request.urlopen(req, timeout=20) as r:
            raw = r.read(3_000_000)
        from PIL import Image
        import numpy as np
        im = Image.open(io.BytesIO(raw)).convert("L")
        return round(float(np.mean(np.asarray(im))), 1)
    except Exception:
        return None


def audit(items):
    recs = {}
    with cf.ThreadPoolExecutor(max_workers=10) as ex:
        fut = {ex.submit(record, it["id"]): it for it in items}
        for f in cf.as_completed(fut):
            it = fut[f]
            try:
                recs[it["id"]] = f.result()
            except Exception as exc:
                recs[it["id"]] = {"id": it["id"], "error": str(exc)[:120]}
    out = []
    with cf.ThreadPoolExecutor(max_workers=10) as ex:
        fut = {ex.submit(brightness, it["thumb"]): it for it in items}
        lum = {}
        for f in cf.as_completed(fut):
            try:
                lum[fut[f]["id"]] = f.result()
            except Exception:
                lum[fut[f]["id"]] = None
    for it in items:
        r = recs[it["id"]]
        text = " ".join(
            [r.get("title", ""), r.get("description", ""), " ".join(r.get("keywords", []))]
        ).lower()
        hits_l = [k for k in LUNAR if k in text]
        hits_c = sorted({k for k in CRATER if k in text})
        hits_d = sorted({k for k in DARK if k in text})
        if r.get("error"):
            verdict, reason = "review", "metadata fetch failed: " + r["error"]
        elif not hits_l:
            verdict, reason = "remove", "not lunar (no moon/lunar in NASA record)"
        elif hits_c or hits_d:
            verdict = "keep"
            reason = "lunar + " + "/".join(
                (["crater:" + ",".join(hits_c)] if hits_c else [])
                + (["dark:" + ",".join(hits_d)] if hits_d else [])
            )
        else:
            verdict, reason = "review", "lunar but no crater/dark token"
        out.append(
            {
                "id": it["id"],
                "title": it["title"],
                "manifest_query": it.get("query", ""),
                "verdict": verdict,
                "reason": reason,
                "mean_luminance": lum.get(it["id"]),
                "nasa_title": r.get("title", ""),
            }
        )
    return out


def main():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    results = audit(manifest["items"])
    apath = MANIFEST_PATH.parent / "library_audit.json"
    apath.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    from collections import Counter
    print(Counter(r["verdict"] for r in results), "-> backend/library_audit.json")
    print("--- REVIEW bucket ---")
    for r in results:
        if r["verdict"] == "review":
            print(f"{r['id']} | {r['title'][:70]} | lum={r['mean_luminance']} | {r['reason']}")
    if "--apply" in sys.argv:
        keep = {r["id"] for r in results if r["verdict"] == "keep"}
        manifest["items"] = [it for it in manifest["items"] if it["id"] in keep]
        MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
        print(f"manifest rewritten: {len(manifest['items'])} kept")


if __name__ == "__main__":
    main()
