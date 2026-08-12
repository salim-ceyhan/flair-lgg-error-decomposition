"""Isaretli halka kontrasti: secim olcutu olarak kullanilabilir mi?

Dondurulmus kanonik TCGA-LGG aday havuzu uzerinde calisir. Mevcut
`ring_contrast` ozniteligi degerlendirme-skoru alaninda bir *orandir*
(ic / dis) ve isaretsizdir. Bu kosu, yogunluk alaninda hesaplanan
*isaretli* farki (ic - dis) ve ilgili varyantlari uretip her birini
etiketsiz bir secim olcutu olarak sinar.

Gercek-referans yalniz geriye donuk puanlamada kullanilir; butun kontrast
varyantlari goruntuden turetilir ve cikarimda erisilebilir.

Gaussian veya medyan on-duzlestirme kullanilmaz (proje kurali).
"""
from __future__ import annotations

# Stable repository paths for package and direct-script execution.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_EVALUATION_DIR = _BootstrapPath(__file__).resolve().parent
_STUDY_ROOT = _EVALUATION_DIR.parent
_PROJECT_ROOT = _STUDY_ROOT.parent
for _bootstrap_path in (_PROJECT_ROOT, _STUDY_ROOT, _EVALUATION_DIR):
    if str(_bootstrap_path) not in _bootstrap_sys.path:
        _bootstrap_sys.path.insert(0, str(_bootstrap_path))
import csv
import hashlib
import json
import random
import statistics as st
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.candidate_selection import persistence as PP  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "tcga_lgg_dataset"
POOL = HERE / "results" / "candidate_pool_acss_canonical"
OUT = HERE / "results" / "signed_ring_contrast"
OUT.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUT / "candidate_contrast_variants.csv"
SUMMARY = OUT / "summary.json"
PROVENANCE = OUT / "provenance.json"

RING_ITERS = 3          # mevcut havuz uretimiyle ayni
SEED = 20260725
BOOT = 20000

FIELDS = [
    "case_id", "candidate_index", "dice", "q_pi",
    "ratio_score", "diff_score",           # degerlendirme-skoru alani
    "ratio_int", "diff_int", "absdiff_int",  # yogunluk alani
    "inside_int", "ring_int",
]


# --------------------------------------------------------------------- veri
def load_rows() -> dict[str, list[dict]]:
    d = defaultdict(list)
    with (POOL / "candidate_features.csv").open(encoding="utf-8", newline="") as h:
        for r in csv.DictReader(h):
            d[r["case_id"]].append(r)
    return {c: sorted(v, key=lambda r: int(r["candidate_index"])) for c, v in d.items()}


def load_masks(case: str) -> list[np.ndarray]:
    z = np.load(POOL / "cases" / f"{case}.npz")
    shape = tuple(map(int, z["image_shape"]))
    n = int(np.prod(shape))
    return [np.unpackbits(x)[:n].reshape(shape).astype(bool) for x in z["packed_masks"]]


def contrast_variants(mask, intensity, evalscore, brain):
    """Bir aday icin butun kontrast varyantlarini dondurur."""
    ring = ndi.binary_dilation(mask, iterations=RING_ITERS) & ~mask & brain
    if not mask.any() or not ring.any():
        return None
    si, so = float(evalscore[mask].mean()), float(evalscore[ring].mean())
    ii, io_ = float(intensity[mask].mean()), float(intensity[ring].mean())
    return dict(
        ratio_score=si / (so + 1e-8), diff_score=si - so,
        ratio_int=ii / (io_ + 1e-8), diff_int=ii - io_,
        absdiff_int=abs(ii - io_), inside_int=ii, ring_int=io_,
    )


def build() -> None:
    rows_by_case = load_rows()
    cases = sorted(rows_by_case)
    with CSV_PATH.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for i, case in enumerate(cases, 1):
            flair = np.load(DATA / case / "flair.npy").astype(float)
            intensity, brain, filt, edge = PP.prep_case(flair)
            evalscore = edge * filt * brain.astype(float)
            masks = load_masks(case)
            rows = rows_by_case[case]
            for k, (m, r) in enumerate(zip(masks, rows)):
                v = contrast_variants(m, intensity, evalscore, brain)
                if v is None:
                    continue
                w.writerow(dict(
                    case_id=case, candidate_index=k,
                    dice=float(r["retrospective_dice"]),
                    q_pi=float(r["canonical_quality"]) * float(r["persistence"]),
                    **{f: round(v[f], 6) for f in v}))
            fh.flush()
            print("[%3d/%d] %s  aday=%d" % (i, len(cases), case, len(masks)), flush=True)


# ----------------------------------------------------------------- deneyler
def load_variants():
    d = defaultdict(list)
    with CSV_PATH.open(encoding="utf-8", newline="") as h:
        for r in csv.DictReader(h):
            rec = {k: float(r[k]) for k in FIELDS if k not in ("case_id", "candidate_index")}
            d[r["case_id"]].append(rec)
    return d


def pctl(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100.0 * (len(xs) - 1))))]


def select(pool, cases, keyfn):
    return {c: (max(pool[c], key=keyfn)["dice"] if pool[c] else 0.0) for c in cases}


def agg(sel):
    xs = list(sel.values())
    return dict(dice=round(st.fmean(xs), 4),
                zeros=sum(1 for x in xs if x == 0.0),
                gt70=sum(1 for x in xs if x > 0.7))


def boot_ci(arm, base, cases):
    rnd = random.Random(SEED)
    diffs = [arm[c] - base[c] for c in cases]
    obs = st.fmean(diffs)
    k = len(diffs)
    reps = sorted(st.fmean([diffs[rnd.randrange(k)] for _ in range(k)]) for _ in range(BOOT))
    return dict(difference=round(obs, 4),
                ci95=[round(reps[int(0.025 * BOOT)], 4), round(reps[int(0.975 * BOOT)], 4)],
                improved=sum(1 for c in cases if arm[c] > base[c] + 1e-9),
                worsened=sum(1 for c in cases if arm[c] < base[c] - 1e-9))


def analyse() -> dict:
    by_case = load_variants()
    cases = sorted(by_case)
    base = select(by_case, cases, lambda r: r["q_pi"])
    res = {"case_count": len(cases),
           "candidate_count": sum(len(v) for v in by_case.values()),
           "baseline_argmax_q_pi": agg(base)}

    # 1) sinif-kosullu dagilimlar
    flat = [r for v in by_case.values() for r in v]
    groups = {"tumor_dice_ge_070": [r for r in flat if r["dice"] >= 0.70],
              "partial": [r for r in flat if 0.05 < r["dice"] < 0.70],
              "distractor_dice_le_005": [r for r in flat if r["dice"] <= 0.05]}
    dist = {}
    for gname, g in groups.items():
        dist[gname] = {"n": len(g)}
        for f in ("ratio_score", "diff_score", "ratio_int", "diff_int", "inside_int", "ring_int"):
            xs = [r[f] for r in g]
            dist[gname][f] = {"median": round(st.median(xs), 4),
                              "iqr": [round(pctl(xs, 25), 4), round(pctl(xs, 75), 4)]}
    res["class_conditional"] = dist

    # 2) tumor adaylarinin vaka-ici yuzdelik konumu (monotonluk sinamasi)
    loc = {}
    for f in ("ratio_score", "diff_int", "ratio_int"):
        pos = []
        for c in cases:
            g = by_case[c]
            vals = sorted(r[f] for r in g)
            for r in g:
                if r["dice"] >= 0.70:
                    pos.append(100.0 * sum(1 for x in vals if x <= r[f]) / len(vals))
        loc[f] = {"median_percentile": round(st.median(pos), 1),
                  "iqr": [round(pctl(pos, 25), 1), round(pctl(pos, 75), 1)]} if pos else None
    res["tumor_percentile_position"] = loc

    # 3) her varyant tek basina secici olarak
    solo = {}
    for f in ("diff_int", "ratio_int", "diff_score", "ratio_score"):
        solo["argmax_" + f] = agg(select(by_case, cases, lambda r, k=f: r[k]))
        solo["argmin_" + f] = agg(select(by_case, cases, lambda r, k=f: -r[k]))
    res["single_criterion"] = solo

    # 4) tek yonlu ve bant on-filtreleri -> argmax Q*pi
    filt = {}
    for f in ("diff_int", "ratio_int"):
        for lo, hi in ((0, 100), (0, 90), (0, 80), (0, 60), (10, 90), (20, 90),
                       (10, 100), (20, 100), (30, 100), (40, 100), (25, 75)):
            pool = {}
            for c in cases:
                g = by_case[c]
                vals = [r[f] for r in g]
                a, b = pctl(vals, lo), pctl(vals, hi)
                keep = [r for r in g if a <= r[f] <= b]
                pool[c] = keep or g
            sel = select(pool, cases, lambda r: r["q_pi"])
            orc = select(pool, cases, lambda r: r["dice"])
            filt["%s_band_%d_%d" % (f, lo, hi)] = {
                **agg(sel), "pool_ceiling": round(st.fmean(orc.values()), 4),
                "median_pool": int(st.median([len(pool[c]) for c in cases]))}
    res["band_filters"] = filt

    # 5) surekli ceza / odul
    pen = {}
    for f, sign in (("diff_int", -1), ("diff_int", +1), ("ratio_int", -1)):
        for gm in (0.25, 0.5, 1.0, 2.0):
            def key(r, k=f, s=sign, g=gm):
                return r["q_pi"] * (1.0 + abs(r[k])) ** (s * g)
            nm = "%s_%s_gamma%.2f" % (f, "penalty" if sign < 0 else "reward", gm)
            pen[nm] = agg(select(by_case, cases, key))
    res["continuous_weighting"] = pen

    # 6) en iyi kolun esli bootstrap GA'si
    best_name, best_key = None, None
    for f, sign in (("diff_int", -1), ("diff_int", +1), ("ratio_int", -1)):
        for gm in (0.25, 0.5, 1.0, 2.0):
            nm = "%s_%s_gamma%.2f" % (f, "penalty" if sign < 0 else "reward", gm)
            if best_name is None or pen[nm]["dice"] > pen[best_name]["dice"]:
                best_name = nm
                best_key = (f, sign, gm)
    f, sign, gm = best_key
    arm = select(by_case, cases, lambda r: r["q_pi"] * (1.0 + abs(r[f])) ** (sign * gm))
    res["best_arm"] = {"name": best_name, **agg(arm), **boot_ci(arm, base, cases),
                       "note": "gamma ayni gelistirme kumesinde tarandi; nokta tahmini iyimserdir"}
    return res


def main() -> None:
    if not CSV_PATH.exists():
        build()
    res = analyse()
    res["ground_truth_policy"] = "GT yalniz geriye donuk puanlamada kullanildi."
    res["gaussian_filtering"] = False
    SUMMARY.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    PROVENANCE.write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pool": str(POOL.relative_to(ROOT)), "data_root": str(DATA.relative_to(ROOT)),
        "ring_iterations": RING_ITERS, "bootstrap": BOOT, "seed": SEED,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
