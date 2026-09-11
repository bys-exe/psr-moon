"""Snapshot the Image Library manifest from paper-derived queries.

Usage (from repo root):
    python backend/build_library.py [--max 300]

Queries NASA's public Image API (no key) with PAPER_QUERIES from
library.py, dedupes by NASA id, HEAD-verifies every thumbnail and full
image URL, and writes backend/library_manifest.json. Re-run anytime to
refresh the snapshot; the app serves the file and never searches live.
"""
import argparse
import concurrent.futures as cf
import datetime as dt
import json
import sys
import urllib.parse
import urllib.request

sys.path.insert(0, ".")
from backend.library import NASA_SEARCH, PAPER_QUERIES, UA  # noqa: E402

PAGE_SIZE = 100


def get_json(url, timeout=30):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def head_ok(url, timeout=15):
    try:
        req = urllib.request.Request(url, headers=UA, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status == 200 and r.headers.get("Content-Type", "").startswith("image/")
    except Exception:
        return False


def search(query, want):
    """Collect up to `want` raw items for one query (paged)."""
    out, page = [], 1
    while len(out) < want:
        qs = urllib.parse.urlencode(
            {"q": query, "media_type": "image", "page": page, "page_size": PAGE_SIZE}
        )
        try:
            data = get_json(f"{NASA_SEARCH}?{qs}")
        except Exception as exc:
            print(f"  ! search failed q={query!r} p={page}: {exc}")
            break
        items = data.get("collection", {}).get("items", [])
        if not items:
            break
        out.extend(items)
        if len(items) < PAGE_SIZE:
            break
        page += 1
    return out[:want]


def build(max_items=300):
    per_query = max(40, max_items // max(1, len(PAPER_QUERIES)))
    seen, cands = set(), []
    for query, rationale in PAPER_QUERIES:
        print(f"* {query!r} ({rationale})")
        for it in search(query, per_query):
            try:
                nasa_id = it["data"][0]["nasa_id"]
            except (KeyError, IndexError):
                continue
            if nasa_id in seen:
                continue
            seen.add(nasa_id)
            links = it.get("links") or []
            thumb = links[0].get("href") if links else None
            if not thumb:
                thumb = f"https://images-assets.nasa.gov/image/{nasa_id}/{nasa_id}~thumb.jpg"
            title = (it["data"][0].get("title") or nasa_id).strip()
            cands.append(
                {
                    "id": nasa_id,
                    "title": title,
                    "source": "NASA Image Library",
                    "query": query,
                    "thumb": thumb,
                    "full_large": f"https://images-assets.nasa.gov/image/{nasa_id}/{nasa_id}~large.jpg",
                    "full_medium": f"https://images-assets.nasa.gov/image/{nasa_id}/{nasa_id}~medium.jpg",
                    "attribution": "Public domain (NASA)",
                }
            )
        print(f"  candidates so far: {len(cands)}")
        if len(cands) >= max_items * 2:
            break

    print(f"* verifying {len(cands)} candidates (thumb + full)...")

    def verify(c):
        if not head_ok(c["thumb"]):
            return None
        full = c["full_large"] if head_ok(c["full_large"]) else None
        if full is None:
            full = c["full_medium"] if head_ok(c["full_medium"]) else None
        if full is None:
            return None
        return {
            "id": c["id"],
            "title": c["title"],
            "source": c["source"],
            "query": c["query"],
            "thumb": c["thumb"],
            "full": full,
            "attribution": c["attribution"],
        }

    items = []
    with cf.ThreadPoolExecutor(max_workers=12) as ex:
        for res in ex.map(verify, cands):
            if res is not None:
                items.append(res)
            if len(items) >= max_items:
                break
    # round-robin trim for query diversity if over cap
    if len(items) > max_items:
        by_q, trimmed = {}, []
        for it in items:
            by_q.setdefault(it["query"], []).append(it)
        while len(trimmed) < max_items and any(by_q.values()):
            for q in list(by_q):
                if by_q[q] and len(trimmed) < max_items:
                    trimmed.append(by_q[q].pop(0))
        items = trimmed

    manifest = {
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "queries": [{"q": q, "rationale": r} for q, r in PAPER_QUERIES],
        "items": items,
    }
    with open("backend/library_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False)
    print(f"* wrote backend/library_manifest.json with {len(items)} items")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=300)
    build(max_items=ap.parse_args().max)
