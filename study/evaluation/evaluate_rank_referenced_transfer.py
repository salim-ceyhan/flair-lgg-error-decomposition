"""Sira-referansli parlaklik kanali kohortlar arasi aktarimi kurtarir mi?

Olculen olgu: dis kohortlarda egitilen yeniden-siralayici TCGA'ya
aktarilamiyor (BraTS 0.499, UCSF-LG 0.604, birlesik 0.583; hepsi etiketsiz
argmax-Q'nun 0.624'unun altinda) ve olculen nedeni oznitelik dagilimi kaymasi
-- en buyuk kayma `mean_score`ta, +2.30 sigma (BraTS) ve +2.15 sigma (UCSF).

Hipotez: `mean_score`u sira-referansli karsiligi `mean_score_rank` ile
degistirmek kaymayi yapisi geregi kaldirir ve aktarimi iyilestirir.

Bu betik ayni modeli, ayni egitim havuzlarini ve ayni degerlendirme
protokolunu iki oznitelik tabaniyla kosar:
  BASE : ... mean_score ...        (kanonik on)
  RANK : ... mean_score_rank ...   (onerilen)

Ayrica her iki tabanda kohort kaymasini (TCGA sigma biriminde) yeniden
hesaplar; hipotezin birincil dogrulama olcutu budur.

GT yalniz egitim hedefi ve puanlama anahtaridir.
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
import argparse, csv, hashlib, json
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
FEAT = HERE / "results" / "rank_referenced_features"
UCSF_GRADES = ROOT / "data" / "ucsf_pdgm_dataset" / "processed" / "grades.csv"
OUT = HERE / "results" / "rank_referenced_transfer"

COMMON = ["log_q", "persistence", "area_frac", "compactness", "solidity",
          "centrality", "contrast", "q_rank_norm", "is_alpha"]
BASES = {"BASE": COMMON + ["mean_score"], "RANK": COMMON + ["mean_score_rank"]}

SEEDS, BASE_SEED, EPOCHS = 12, 20260726, 300
VIABLE_TAU, GATE_W, ZERO_TOL = 0.10, 1.0, 1e-9


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Dropout(0.3),
                                   nn.Linear(32, 16), nn.ReLU())
        self.dice = nn.Linear(16, 1)
        self.viable = nn.Linear(16, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.dice(h).squeeze(-1), self.viable(h).squeeze(-1)


def load(tag, features):
    path = FEAT / tag / "candidate_features.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    cases = sorted({r["case"] for r in rows})
    by = {c: [] for c in cases}
    for r in rows:
        by[r["case"]].append(r)
    X = {c: np.array([[float(r[f]) for f in features] for r in by[c]], np.float32)
         for c in cases}
    Y = {c: np.array([float(r["dice"]) for r in by[c]], np.float32) for c in cases}
    return cases, X, Y


def ucsf_lower_grade():
    if not UCSF_GRADES.exists():
        return set()
    return {r["case"].strip() for r in csv.DictReader(UCSF_GRADES.open(encoding="utf-8"))
            if r["grade"].strip() in {"2", "3"}}


def train(Xtr, ytr, seed, d):
    torch.manual_seed(seed)
    model = MLP(d)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-3)
    mse, bce = nn.MSELoss(), nn.BCEWithLogitsLoss()
    viable = (ytr > VIABLE_TAU).float()
    model.train()
    for _ in range(EPOCHS):
        opt.zero_grad()
        pred, logit = model(Xtr)
        (mse(pred, ytr) + GATE_W * bce(logit, viable)).backward()
        opt.step()
    model.eval()
    return model


def evaluate(pool_cases, Xp, Yp, te_cases, Xt, Yt, d, seeds=SEEDS):
    Xtr_np = np.stack([Xp[c] for c in pool_cases])
    ytr_np = np.stack([Yp[c] for c in pool_cases])
    flat = Xtr_np.reshape(-1, d)
    mu, sd = flat.mean(0), flat.std(0) + 1e-9
    Xtr = torch.tensor((Xtr_np - mu) / sd)
    ytr = torch.tensor(ytr_np)
    Xte = {c: torch.tensor((Xt[c] - mu) / sd) for c in te_cases}

    per_seed, ranks = [], {c: [] for c in te_cases}
    for s in range(seeds):
        model = train(Xtr, ytr, BASE_SEED + s, d)
        picks = []
        for c in te_cases:
            with torch.no_grad():
                pred, _ = model(Xte[c])
            score = pred.numpy()
            ranks[c].append(np.argsort(np.argsort(score)))
            picks.append(Yt[c][int(np.argmax(score))])
        per_seed.append(np.array(picks))
    ens = np.array([Yt[c][int(np.argmax(np.mean(ranks[c], 0)))] for c in te_cases])
    m = [float(v.mean()) for v in per_seed]
    z = [int((v <= ZERO_TOL).sum()) for v in per_seed]
    return {"train_patients": len(pool_cases),
            "per_seed_mean_dice": round(float(np.mean(m)), 4),
            "per_seed_sd": round(float(np.std(m)), 4),
            "per_seed_zeros": round(float(np.mean(z)), 2),
            "ensemble_mean_dice": round(float(ens.mean()), 4),
            "ensemble_zeros": int((ens <= ZERO_TOL).sum())}


def shift(Xt, te_cases, Xp, pool_cases, features):
    """Egitim havuzunun TCGA'ya gore ortalama kaymasi, TCGA sigma biriminde."""
    t = np.concatenate([Xt[c] for c in te_cases], 0)
    p = np.concatenate([Xp[c] for c in pool_cases], 0)
    mu, sd = t.mean(0), t.std(0) + 1e-9
    z = (p.mean(0) - mu) / sd
    return {f: round(float(v), 3) for f, v in zip(features, z)}, round(
        float(np.linalg.norm(z)), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=SEEDS)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    lg = ucsf_lower_grade()
    res = {"seeds": args.seeds, "feature_bases": BASES, "arms": {}}

    for basis, feats in BASES.items():
        te, Xt, Yt = load("tcga", feats)
        bc, Xb, Yb = load("brats", feats)
        uc, Xu, Yu = load("ucsf", feats)
        ulg = [c for c in uc if c in lg] or uc

        pools = {"brats": (bc, Xb, Yb), "ucsf_lg": (ulg, Xu, Yu), "ucsf": (uc, Xu, Yu)}
        arm = {"tcga_cases": len(te), "pool_sizes": {k: len(v[0]) for k, v in pools.items()}}
        for name, (pc, Xp, Yp) in pools.items():
            arm[name] = evaluate(pc, Xp, Yp, te, Xt, Yt, len(feats), args.seeds)
            zz, l2 = shift(Xt, te, Xp, pc, feats)
            arm[name]["cohort_shift_z"] = zz
            arm[name]["cohort_shift_l2"] = l2
        res["arms"][basis] = arm
        print(json.dumps({basis: arm}, indent=2), flush=True)

    # birincil olcut: parlaklik kanalinin kaymasi
    res["primary_readout"] = {
        pool: {
            "BASE_mean_score_z": res["arms"]["BASE"][pool]["cohort_shift_z"]["mean_score"],
            "RANK_mean_score_rank_z":
                res["arms"]["RANK"][pool]["cohort_shift_z"]["mean_score_rank"],
            "BASE_transfer_dice": res["arms"]["BASE"][pool]["ensemble_mean_dice"],
            "RANK_transfer_dice": res["arms"]["RANK"][pool]["ensemble_mean_dice"],
        } for pool in ("brats", "ucsf_lg", "ucsf")}
    res["reference_points"] = {"argmax_Q": 0.624, "within_tcga_crossfit": 0.678,
                               "insample_bound": 0.753, "shortlist_oracle": 0.778}
    res["notes"] = ("Sira ekseni pencere kenarlari ayarlanmamistir. "
                    "GT yalniz egitim hedefi ve puanlama. Gaussian yok.")
    (OUT / "summary.json").write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n",
                                      encoding="utf-8")
    (OUT / "provenance.json").write_text(json.dumps({
        "features_root": str(FEAT.relative_to(ROOT)), "base_seed": BASE_SEED,
        "epochs": EPOCHS,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        indent=2) + "\n", encoding="utf-8")
    print(json.dumps(res["primary_readout"], indent=2))


if __name__ == "__main__":
    main()
