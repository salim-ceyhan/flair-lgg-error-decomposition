"""Sira-referansli kalite olcutu: Q'nun parlaklik kanalini mutlak olcekten cikarmak.

Kanonik Q = A * s_bar * cp * sol; burada s_bar maske-ici ortalama skordur ve
mutlak yogunluk olceginde tanimlidir. Bu kosu s_bar'i beyin-ROI dagilimina
gore sira-donusturulmus karsiligiyla degistirir ve ayni dondurulmus havuzda
secim basarimini olcer. Uretim degismez, dolayisiyla havuz tavani sabittir.

Gercek-referans yalniz geriye donuk puanlamada kullanilir. Gaussian yok.
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
import csv, hashlib, json, random, statistics as st, sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.candidate_selection import persistence as PP  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "tcga_lgg_dataset"
POOL = HERE / "results" / "candidate_pool_acss_canonical"
OUT = HERE / "results" / "rank_referenced_quality"
OUT.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUT / "candidate_rank_features.csv"
SEED, BOOT = 20260725, 20000

TAU_LO, TAU_HI_STRICT, TAU_HI_LOOSE, SW = 0.28, 0.82, 0.97, 0.05


def load_rows():
    d = defaultdict(list)
    with (POOL / "candidate_features.csv").open(encoding="utf-8", newline="") as h:
        for r in csv.DictReader(h):
            d[r["case_id"]].append(r)
    return {c: sorted(v, key=lambda r: int(r["candidate_index"])) for c, v in d.items()}


def load_masks(case):
    z = np.load(POOL / "cases" / f"{case}.npz")
    shape = tuple(map(int, z["image_shape"]))
    n = int(np.prod(shape))
    return [np.unpackbits(x)[:n].reshape(shape).astype(bool) for x in z["packed_masks"]]


def rank_map(field, brain):
    """Beyin-ici degerleri [0,1] nicelik siralarina donusturur."""
    out = np.zeros_like(field, dtype=float)
    v = field[brain]
    order = np.argsort(np.argsort(v, kind="mergesort"), kind="mergesort")
    out[brain] = order / max(len(v) - 1, 1)
    return out


def sigmoid_band(x, lo, hi, s):
    return 1.0 / (1.0 + np.exp(-(x - lo) / s)) * 1.0 / (1.0 + np.exp((x - hi) / s))


def build():
    rows_by_case = load_rows()
    cases = sorted(rows_by_case)
    fields = ["case_id", "candidate_index", "dice", "q_canon", "persistence",
              "area", "compactness", "solidity",
              "sbar_abs", "sbar_rank", "sbar_winrank"]
    with CSV_PATH.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for i, case in enumerate(cases, 1):
            flair = np.load(DATA / case / "flair.npy").astype(float)
            intensity, brain, filt, edge = PP.prep_case(flair)
            evalscore = edge * filt * brain.astype(float)

            # (a) skor alaninin sira-donusumu
            eval_rank = rank_map(evalscore, brain)
            # (b) yogunlugun sira-donusumu uzerine ayni bant -> sira-referansli win
            int_rank = rank_map(intensity, brain)
            win_rank = sigmoid_band(int_rank, TAU_LO, TAU_HI_LOOSE, SW)
            score_winrank = edge * filt * brain.astype(float) * win_rank

            for k, (m, r) in enumerate(zip(load_masks(case), rows_by_case[case])):
                if not m.any():
                    continue
                w.writerow(dict(
                    case_id=case, candidate_index=k,
                    dice=float(r["retrospective_dice"]),
                    q_canon=float(r["canonical_quality"]),
                    persistence=float(r["persistence"]),
                    area=float(r["area_px"]),
                    compactness=float(r["compactness"]),
                    solidity=float(r["solidity"]),
                    sbar_abs=round(float(evalscore[m].mean()), 6),
                    sbar_rank=round(float(eval_rank[m].mean()), 6),
                    sbar_winrank=round(float(score_winrank[m].mean()), 6)))
            fh.flush()
            print("[%3d/%d] %s" % (i, len(cases), case), flush=True)


def analyse():
    by_case = defaultdict(list)
    with CSV_PATH.open(encoding="utf-8", newline="") as h:
        for r in csv.DictReader(h):
            by_case[r["case_id"]].append({k: (v if k == "case_id" else float(v))
                                          for k, v in r.items()})
    cases = sorted(by_case)

    def sel(key):
        return {c: max(by_case[c], key=key)["dice"] for c in cases}

    def agg(s):
        x = list(s.values())
        return dict(dice=round(st.fmean(x), 4), zeros=sum(1 for v in x if v == 0),
                    gt70=sum(1 for v in x if v > 0.7))

    def boot(a, b):
        rnd = random.Random(SEED)
        d = [a[c] - b[c] for c in cases]
        k = len(d)
        reps = sorted(st.fmean([d[rnd.randrange(k)] for _ in range(k)]) for _ in range(BOOT))
        return dict(difference=round(st.fmean(d), 4),
                    ci95=[round(reps[int(.025 * BOOT)], 4), round(reps[int(.975 * BOOT)], 4)],
                    improved=sum(1 for c in cases if a[c] > b[c] + 1e-9),
                    worsened=sum(1 for c in cases if a[c] < b[c] - 1e-9))

    def q_of(r, sfield):
        return r["area"] * r[sfield] * r["compactness"] * r["solidity"]

    base = sel(lambda r: r["q_canon"] * r["persistence"])
    res = {"case_count": len(cases),
           "candidate_count": sum(len(v) for v in by_case.values()),
           "baseline_canonical": agg(base),
           "ceiling_unchanged": round(st.fmean([max(r["dice"] for r in by_case[c])
                                                for c in cases]), 4)}
    arms = {}
    for nm, sf in (("sbar_abs_rebuilt", "sbar_abs"),
                   ("sbar_rank", "sbar_rank"),
                   ("sbar_winrank", "sbar_winrank")):
        for beta, tag in ((1.0, "pi1"), (0.0, "pi0")):
            key = lambda r, f=sf, b=beta: q_of(r, f) * (r["persistence"] ** b)
            s = sel(key)
            arms["%s_%s" % (nm, tag)] = {**agg(s), **boot(s, base)}
    res["arms"] = arms

    # vaka-ici Spearman: s_bar varyantlarinin siralama gucu
    try:
        from scipy.stats import spearmanr
        rho = {}
        for sf in ("sbar_abs", "sbar_rank", "sbar_winrank"):
            v = []
            for c in cases:
                g = by_case[c]
                if len(g) > 3:
                    x = spearmanr([r[sf] for r in g], [r["dice"] for r in g]).statistic
                    if np.isfinite(x):
                        v.append(float(x))
            rho[sf] = {"median": round(st.median(v), 4),
                       "positive_fraction": round(sum(1 for x in v if x > 0) / len(v), 3)}
        res["within_case_spearman_vs_dice"] = rho
    except Exception as exc:  # pragma: no cover
        res["within_case_spearman_vs_dice"] = {"error": str(exc)}

    res["scope"] = ("Yalniz secim tarafi sinanmistir; aday uretimi degismedigi icin "
                    "havuz tavani sabittir.")
    res["ground_truth_policy"] = "GT yalniz geriye donuk puanlamada."
    res["gaussian_filtering"] = False
    return res


def main():
    if not CSV_PATH.exists():
        build()
    res = analyse()
    (OUT / "summary.json").write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n",
                                      encoding="utf-8")
    (OUT / "provenance.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pool": str(POOL.relative_to(ROOT)), "seed": SEED, "bootstrap": BOOT,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        indent=2) + "\n", encoding="utf-8")
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
