"""Curate the library to moon-craters / moon-dark-regions ONLY.

Usage (from repo root):
    python backend/refine_library.py

Phase A (purge): drops ground/illustration/non-lunar frames by ID prefix
  (KSC/ARC/SSC/NHQ/CEB/JSC center photos = clean rooms, launches, events,
  labs) plus explicit IDs verified by description spot-checks
  (astronaut EVA, Earthrise, spacecraft illustrations, hardware, maps,
  Europa, ...). Every drop reason is printed.

Phase B (top up, moon-only): searches crater/dark oriented queries, then
  keeps a candidate ONLY if its NASA record is lunar + crater/dark AND
  contains no exclusion token (crew, hardware, launches, illustrations,
  maps, other planets...). Survivors are HEAD-verified like the builder.

Phase C (confirm): re-audits the whole manifest; asserts every verdict
  is keep, and prints the final count.
"""
import json
import re
import sys

sys.path.insert(0, ".")
from backend.audit_library import audit  # noqa: E402
from backend.build_library import head_ok, search  # noqa: E402
from backend.library import MANIFEST_PATH  # noqa: E402

# Ground-facility prefixes: clean rooms, launches, press events, labs.
REMOVE_PREFIXES = ("KSC-", "NHQ", "CEB_", "ARC-", "SSC-", "jsc2024", "jsc2025")

# Verified by description spot-check (astronaut EVA, Earthrise, illustrations,
# hardware, maps, non-lunar bodies).
REMOVE_IDS = {
    "as16-116-18671", "as14-64-9099", "as14-68-9453", "as14-64-9129",
    "as08-13-2329", "as08-14-2384", "as08-14-2383",
    "ACD22-0003-001", "ACD22-0003-002",
    "PIA25626", "PIA25258", "PIA25257",
    "PIA12925", "PIA12088", "PIA18163", "PIA12072",
    "PIA25330", "GSFC_20171208_Archive_e001048",
}

# (query, rationale) — all moon-only by construction.
NEW_QUERIES = [
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

EXCLUDE = [
    "astronaut", "cosmonaut", "crew", "eva", "spacewalk", "engineer",
    "technician", "facility", "launch", "rocket", "aircraft", "illustration",
    "artist", "portrait", "conference", "ceremony", "laboratory",
    "spacesuit", "map", "diagram", "europa", "enceladus", "titan",
    "ganymede", "callisto", "mars", "jupiter", "saturn", "venus",
    "mercury", "asteroid", "comet", "juno", "cassini", "earthrise",
]
EXCLUDE_RE = re.compile(r"\b(" + "|".join(EXCLUDE) + r")\b")


def drop_reason(item_id, title):
    if any(item_id.startswith(p) for p in REMOVE_PREFIXES):
        return "ground-facility photo (prefix)"
    if item_id in REMOVE_IDS:
        return "non crater/dark subject (spot-checked)"
    return None


def main():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    items = manifest["items"]
    print(f"starting: {len(items)}")

    # ---- Phase A: purge ----
    kept, dropped = [], []
    for it in items:
        reason = drop_reason(it["id"], it["title"])
        if reason:
            dropped.append((it["id"], it["title"][:60], reason))
        else:
            kept.append(it)
    print(f"purged {len(dropped)}, kept {len(kept)}")
    for did, title, reason in dropped:
        print(f"  - {did} | {title} | {reason}")

    seen = {it["id"] for it in kept}

    # ---- Phase B: top up ----
    cands = []
    for query, rationale in NEW_QUERIES:
        n0 = len(cands)
        for it in search(query, 100):
            try:
                nid = it["data"][0]["nasa_id"]
            except (KeyError, IndexError):
                continue
            if nid in seen:
                continue
            seen.add(nid)
            links = it.get("links") or []
            thumb = (
                links[0].get("href")
                if links
                else f"https://images-assets.nasa.gov/image/{nid}/{nid}~thumb.jpg"
            )
            title = (it["data"][0].get("title") or nid).strip()
            cands.append({"id": nid, "title": title, "query": query,
                          "thumb": thumb})
        print(f"* {query!r}: +{len(cands) - n0} new candidates")

    # HEAD-verify thumbs first (cheap), then audit text, then verify fulls.
    print(f"* HEAD-verifying {len(cands)} thumbs...")
    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(max_workers=12) as ex:
        ok = list(ex.map(lambda c: head_ok(c["thumb"]), cands))
    cands = [c for c, good in zip(cands, ok) if good]
    print(f"  thumbs ok: {len(cands)}")

    verdicts = {r["id"]: r for r in audit(cands)}
    survivors = []
    for c in cands:
        v = verdicts[c["id"]]
        if v["verdict"] != "keep":
            continue
        # exclusion pass over the NASA record is implicit in verdict text;
        # re-check title+query text for subject tokens here:
        text = f"{v['nasa_title']} {v['title']}".lower()
        if EXCLUDE_RE.search(text):
            continue
        if text.strip().startswith("earth"):
            continue
        survivors.append(c)
    print(f"  audit+exclusion survivors: {len(survivors)}")

    def full_for(c):
        large = c["thumb"].rsplit("~", 1)[0] + "~large.jpg"
        medium = c["thumb"].rsplit("~", 1)[0] + "~medium.jpg"
        if head_ok(large):
            return large
        if head_ok(medium):
            return medium
        return None

    added = []
    with cf.ThreadPoolExecutor(max_workers=12) as ex:
        for c, full in zip(survivors, ex.map(full_for, survivors)):
            if full is None:
                continue
            added.append({
                "id": c["id"], "title": c["title"],
                "source": "NASA Image Library", "query": c["query"],
                "thumb": c["thumb"], "full": full,
                "attribution": "Public domain (NASA)",
            })
    print(f"  added: {len(added)}")
    kept.extend(added)

    # ---- Phase C: confirm whole manifest ----
    import datetime as dt
    final = {r["id"]: r for r in audit(kept)}
    from collections import Counter
    print("final verdicts:", Counter(r["verdict"] for r in final.values()))
    bad = [r for r in final.values() if r["verdict"] != "keep"]
    for r in bad:
        print("  !! NOT-KEEP:", r["id"], "|", r["title"][:70], "|", r["reason"])
    assert not bad, f"{len(bad)} items failed confirmation"

    manifest["items"] = kept
    manifest["generated"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    manifest["curation"] = (
        "Moon craters and moon dark regions only. Ground-facility, crew, "
        "spacecraft-illustration, hardware, map and non-lunar frames removed; "
        "every item confirmed against its NASA record."
    )
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    print(f"manifest rewritten: {len(kept)} items, ALL confirmed keep")


if __name__ == "__main__":
    main()
