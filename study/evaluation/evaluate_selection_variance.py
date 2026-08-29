"""Katman 1'in sapmasi ile Katman 2'nin varyansini ayri ayri olcer.

Uc olcum:

  (A) Kapali-formlu secimin kararliligi. Kanonik havuzda Q*pi puanina
      carpimsal gurultu eklenir; argmax'in kac vakada degistigi sayilir.
      Kararli cikmasi, Katman 1 hatasinin varyans degil *sapma* oldugunu
      gosterir.

  (B) Sapmanin buyuklugu. Top-60 kisa listesinde argmax-Q ile kisa liste
      tavani arasindaki fark.

  (C) Ogrenilmis siralayicinin tohumdan tohuma kararsizligi ve mekanizmasi.
      Hasta duzeyi 5-katli capraz-uyarlama, 12 tohum; her tohumun her
      vakadaki secimi kaydedilir. Kararsiz vakalarda en iyi adayin
      standartlastirilmis oznitelik uzayindaki en yakin sifir-Dice komsusu
      ile vaka-ici tipik aday uzakligi karsilastirilir.

GT yalniz geriye donuk puanlama ve egitim hedefi olarak kullanilir.
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
import csv, hashlib, json, statistics as st
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
POOL = HERE / "results" / "candidate_pool_acss_canonical" / "candidate_features.csv"
TOP60 = HERE / "results" / "supervised_topk_reranker" / "top60" / "candidate_features.csv"
OUT = HERE / "results" / "selection_variance"

FEATURES = ["log_q", "persistence", "area_frac", "compactness", "solidity",
            "centrality", "contrast", "q_rank_norm", "is_alpha", "mean_score"]
NOISE = (0.0, 0.005, 0.01, 0.02, 0.05)
DRAWS = 400
NOISE_SEED = 20260725
FOLD_SEED = 20260722
TORCH_SEED = 20260726
SEEDS, FOLDS, EPOCHS, TAU = 12, 5, 300, 0.10
BIAS_GAP = 0.30
UNSTABLE_SD = 0.10
MATERIAL_SD = 0.05
GT_MIN_AREA = 582
AREA_FLOORS = (0, 200, 300, 400, 500, 582, 700, 900, 1200, 1600)


# --------------------------------------------------------------- (A) sarsma
def perturbation_test() -> dict:
    by = defaultdict(list)
    for r in csv.DictReader(POOL.open(encoding="utf-8")):
        by[r["case_id"]].append((
            float(r["canonical_quality"]) * float(r["persistence"]),
            float(r["retrospective_dice"])))
    cases = sorted(by)
    rng = np.random.default_rng(NOISE_SEED)
    out = {"case_count": len(cases), "draws": DRAWS, "levels": {}}
    for eps in NOISE:
        means, moved, material, sds = [], 0, 0, []
        for c in cases:
            q = np.array([x[0] for x in by[c]])
            d = np.array([x[1] for x in by[c]])
            s = np.log(np.maximum(q, 1e-300))
            if eps == 0.0:
                means.append(float(d[int(np.argmax(s))]))
                sds.append(0.0)
                continue
            scale = s.std() if s.std() > 0 else 1.0
            picks = d[np.argmax(s[None, :] + rng.normal(0, eps * scale,
                                                        (DRAWS, len(s))), axis=1)]
            means.append(float(picks.mean()))
            sds.append(float(picks.std()))
            moved += int(picks.std() > 0)
            material += int(picks.std() > MATERIAL_SD)
        out["levels"]["%.3f" % eps] = {
            "mean_dice": round(st.fmean(means), 4),
            "cases_moving_at_all": int(moved),
            "cases_moving_materially": int(material),
            "mean_per_case_dice_sd": round(st.fmean(sds), 4)}
    out["material_sd_threshold"] = MATERIAL_SD
    out["note"] = ("Cogu vakada secim oynar fakat Dice'i degistirmez; "
                   "maddi oynama olcutu vaka-duzeyi Dice sd > %.2f." % MATERIAL_SD)
    return out


# --------------------------------------------------- (D) en kucuk-alan tabani
def area_floor_test() -> dict:
    """Celdirici ikizler bir asgari-alan kosuluyla elenebilir mi?"""
    by = defaultdict(list)
    for r in csv.DictReader(POOL.open(encoding="utf-8")):
        by[r["case_id"]].append(dict(
            q=float(r["canonical_quality"]) * float(r["persistence"]),
            d=float(r["retrospective_dice"]), a=float(r["area_px"])))
    cases = sorted(by)

    def pick(T):
        return {c: max([r for r in by[c] if r["a"] >= T] or by[c],
                       key=lambda r: r["q"])["d"] for c in cases}

    def ceiling(T):
        return st.fmean([max(r["d"] for r in ([x for x in by[c] if x["a"] >= T] or by[c]))
                         for c in cases])

    base = pick(0)
    rnd_seed = NOISE_SEED
    out = {"gt_min_area_px": GT_MIN_AREA, "levels": {}}
    for T in AREA_FLOORS:
        s = pick(T)
        row = {"mean_dice": round(st.fmean(s.values()), 4),
               "zeros": sum(1 for v in s.values() if v == 0),
               "pool_ceiling": round(ceiling(T), 4)}
        if T:
            import random as _rnd
            r = _rnd.Random(rnd_seed)
            diffs = [s[c] - base[c] for c in cases]
            k = len(diffs)
            reps = sorted(st.fmean([diffs[r.randrange(k)] for _ in range(k)])
                          for _ in range(20000))
            row.update(difference=round(st.fmean(diffs), 4),
                       ci95=[round(reps[500], 4), round(reps[19500], 4)],
                       improved=sum(1 for c in cases if s[c] > base[c] + 1e-9),
                       worsened=sum(1 for c in cases if s[c] < base[c] - 1e-9))
        out["levels"][str(T)] = row
    out["verdict"] = ("Hicbir esikte GA sifiri dislamaz; ayrica en iyi esik "
                      "degerlendirme kohortunun gercek-referansindan turetildigi "
                      "icin etiketsiz cikarimda kullanilamaz.")
    return out


# ----------------------------------------------------------- (B)+(C) veri
def load_top60():
    by = defaultdict(list)
    for r in csv.DictReader(TOP60.open(encoding="utf-8")):
        by[r["case"]].append(r)
    cases = sorted(by)
    for c in cases:
        by[c].sort(key=lambda r: float(r["q_rank_norm"]))
    X = {c: np.array([[float(r[f]) for f in FEATURES] for r in by[c]], np.float32)
         for c in cases}
    Y = {c: np.array([float(r["dice"]) for r in by[c]], np.float32) for c in cases}
    return cases, X, Y


def bias_magnitude(cases, Y) -> dict:
    gaps = {c: float(Y[c].max() - Y[c][0]) for c in cases}
    big = sorted((g, c) for c, g in gaps.items() if g >= BIAS_GAP)
    return {"threshold": BIAS_GAP,
            "cases_with_gap": len(big),
            "case_count": len(cases),
            "median_gap_among_them": round(st.median([g for g, _ in big]), 3) if big else None,
            "largest": [{"case": c, "argmax_q_dice": round(float(Y[c][0]), 3),
                         "shortlist_ceiling": round(float(Y[c].max()), 3)}
                        for g, c in sorted(big, reverse=True)[:5]]}


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.t = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Dropout(0.3),
                               nn.Linear(32, 16), nn.ReLU())
        self.d = nn.Linear(16, 1)
        self.v = nn.Linear(16, 1)

    def forward(self, x):
        h = self.t(x)
        return self.d(h).squeeze(-1), self.v(h).squeeze(-1)


def seed_instability(cases, X, Y) -> tuple[dict, dict]:
    rng = np.random.default_rng(FOLD_SEED)
    order = list(cases)
    rng.shuffle(order)
    folds = [order[i::FOLDS] for i in range(FOLDS)]
    picks = {c: [] for c in cases}
    for s in range(SEEDS):
        for f in range(FOLDS):
            te = folds[f]
            tr = [c for c in cases if c not in te]
            Xtr = np.stack([X[c] for c in tr])
            ytr = np.stack([Y[c] for c in tr])
            flat = Xtr.reshape(-1, len(FEATURES))
            mu, sd = flat.mean(0), flat.std(0) + 1e-9
            xt, yt = torch.tensor((Xtr - mu) / sd), torch.tensor(ytr)
            torch.manual_seed(TORCH_SEED + s)
            m = MLP(len(FEATURES))
            opt = torch.optim.Adam(m.parameters(), lr=2e-3, weight_decay=1e-3)
            mse, bce = nn.MSELoss(), nn.BCEWithLogitsLoss()
            vi = (yt > TAU).float()
            m.train()
            for _ in range(EPOCHS):
                opt.zero_grad()
                p, l = m(xt)
                (mse(p, yt) + bce(l, vi)).backward()
                opt.step()
            m.eval()
            for c in te:
                with torch.no_grad():
                    p, _ = m(torch.tensor((X[c] - mu) / sd))
                picks[c].append(float(Y[c][int(np.argmax(p.numpy()))]))
        print("tohum %d/%d" % (s + 1, SEEDS), flush=True)

    sds = {c: st.pstdev(picks[c]) for c in cases}
    per_seed_zero = [sum(1 for c in cases if picks[c][i] <= 1e-9) for i in range(SEEDS)]
    res = {
        "seeds": SEEDS, "folds": FOLDS,
        "per_seed_mean_dice": round(st.fmean(
            [st.fmean([picks[c][i] for c in cases]) for i in range(SEEDS)]), 4),
        "per_seed_zero_counts": per_seed_zero,
        "cases_identical_across_all_seeds": sum(1 for c in cases if sds[c] == 0.0),
        "cases_sd_above_%.2f" % UNSTABLE_SD: sum(1 for c in cases if sds[c] > UNSTABLE_SD),
        "cases_flipping_zero_to_above_0.5": sum(
            1 for c in cases
            if any(x <= 1e-9 for x in picks[c]) and max(picks[c]) > 0.5),
        "most_unstable": [
            {"case": c, "mean": round(st.fmean(picks[c]), 3), "sd": round(sds[c], 3),
             "zero_seeds": sum(1 for x in picks[c] if x <= 1e-9),
             "min": round(min(picks[c]), 2), "max": round(max(picks[c]), 2)}
            for c in sorted(cases, key=lambda x: -sds[x])[:6]],
    }
    return res, sds


def twin_mechanism(cases, X, Y, sds) -> dict:
    allX = np.concatenate([X[c] for c in cases], 0)
    scale = allX.std(0) + 1e-9
    rows = []
    for c in sorted(cases, key=lambda x: -sds[x])[:6]:
        d = Y[c]
        z = [i for i in range(len(d)) if d[i] <= 1e-9]
        if not z:
            continue
        Xc = X[c] / scale
        bi = int(np.argmax(d))
        dist, j = min((float(np.linalg.norm(Xc[bi] - Xc[i])), i) for i in z)
        pw = [float(np.linalg.norm(Xc[a] - Xc[b]))
              for a in range(0, len(d), 7) for b in range(0, len(d), 7) if a < b]
        rows.append({"case": c,
                     "dice_gap": round(float(d[bi] - d[j]), 3),
                     "nearest_zero_distance": round(dist, 3),
                     "median_within_case_distance": round(st.median(pw), 3)})
    return {"note": "standartlastirilmis oznitelik uzayinda Oklid uzakligi",
            "cases": rows,
            "nearest_zero_range": [round(min(r["nearest_zero_distance"] for r in rows), 2),
                                   round(max(r["nearest_zero_distance"] for r in rows), 2)],
            "typical_range": [round(min(r["median_within_case_distance"] for r in rows), 2),
                              round(max(r["median_within_case_distance"] for r in rows), 2)]}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cases, X, Y = load_top60()
    res = {"layer1_perturbation": perturbation_test(),
           "layer1_bias_magnitude": bias_magnitude(cases, Y),
           "layer1_area_floor": area_floor_test()}
    inst, sds = seed_instability(cases, X, Y)
    res["layer2_seed_instability"] = inst
    res["layer2_twin_mechanism"] = twin_mechanism(cases, X, Y, sds)
    res["interpretation"] = ("Katman 1 kararlidir ve hatasi sapmadir; "
                             "Katman 2'nin hatasi varyanstir.")
    res["ground_truth_policy"] = "GT yalniz geriye donuk puanlama ve egitim hedefi."
    res["gaussian_filtering"] = False
    (OUT / "summary.json").write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n",
                                      encoding="utf-8")
    (OUT / "provenance.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pool": str(POOL.relative_to(ROOT)), "shortlist": str(TOP60.relative_to(ROOT)),
        "noise_seed": NOISE_SEED, "fold_seed": FOLD_SEED, "torch_seed": TORCH_SEED,
        "draws": DRAWS, "epochs": EPOCHS,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        indent=2) + "\n", encoding="utf-8")
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
