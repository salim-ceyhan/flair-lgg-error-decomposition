"""Kanal ablasyonu: yalniz alpha-kesit, yalniz traversal, ikisi birlikte.

Makalede iki uretim kanali birlikte kullaniliyor ama 'yalniz alpha-kesit
kullansaydik ne olurdu' karsilastirmasi yapilmamisti. Bu, mimarinin
gerekceleendirilmesi icin zorunlu bir ablasyondur: eger alpha kolu tek basina
birlesik hattin yaptigini yapiyorsa, traversal makinesinin tasidigi yuk
sorgulanmalidir.

Karsilastirma tam havuz uzerinde, ayni etiketsiz kalite olcutuyle yapilir:
her kanal icin havuz o kanalin adaylariyla sinirlanir, argmax-Q secilir ve
tum-vaka Dice ile tam-basarisizlik sayisi raporlanir. Referans-ust-sinir
(oracle) satirlari kanalin uretim kapasitesini gosterir, cikarim degildir.
"""
from __future__ import annotations
import csv, json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
POOL = HERE / "results" / "full_pool_quality_components" / "tcga" / "pool.csv"
OUT = HERE / "results" / "channel_ablation"
ZERO_TOL = 1e-9
SEED = 20260726


def boot(delta, n=20000):
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(delta), (n, len(delta)))
    return [float(x) for x in np.quantile(delta[idx].mean(1), [.025, .975])]


def sign_test(delta):
    from math import comb
    pos = int((delta > 1e-12).sum()); neg = int((delta < -1e-12).sum())
    n = pos + neg
    if n == 0:
        return 1.0, pos, neg
    k = min(pos, neg)
    p = sum(comb(n, i) for i in range(k + 1)) / 2 ** n * 2
    return min(1.0, p), pos, neg


def main():
    rows = list(csv.DictReader(POOL.open(encoding="utf-8")))
    by = {}
    for r in rows:
        by.setdefault(r["case"], []).append(r)
    cases = sorted(by)

    CH = {
        "yalnız traversal (kanal A)": lambda r: float(r["is_alpha"]) == 0.0,
        "yalnız alpha-kesit (kanal B)": lambda r: float(r["is_alpha"]) == 1.0,
        "birleşik (kanonik havuz)": lambda r: True,
    }

    sel, orc, sizes = {}, {}, {}
    for name, keep in CH.items():
        s_, o_, n_ = [], [], []
        for c in cases:
            rs = [r for r in by[c] if keep(r)]
            if not rs:
                s_.append(0.0); o_.append(0.0); n_.append(0)
                continue
            q = np.array([float(r["quality"]) for r in rs])
            dc = np.array([float(r["dice"]) for r in rs])
            s_.append(float(dc[int(np.argmax(q))]))
            o_.append(float(dc.max()))
            n_.append(len(rs))
        sel[name] = np.array(s_); orc[name] = np.array(o_); sizes[name] = np.array(n_)

    base = "birleşik (kanonik havuz)"
    print(f"{'kanal':30s} {'aday/vaka':>10s}  {'argmax-Q':>9s} {'sıfır':>6s}  "
          f"{'oracle':>8s} {'sıfır':>6s}")
    summary = {}
    for name in CH:
        s_, o_ = sel[name], orc[name]
        print(f"{name:30s} {int(np.median(sizes[name])):10d}  "
              f"{s_.mean():9.4f} {int((s_ <= ZERO_TOL).sum()):6d}  "
              f"{o_.mean():8.4f} {int((o_ <= ZERO_TOL).sum()):6d}")
        summary[name] = {
            "median_candidates": int(np.median(sizes[name])),
            "selected_mean_dice": float(s_.mean()),
            "selected_zeros": int((s_ <= ZERO_TOL).sum()),
            "oracle_mean_dice": float(o_.mean()),
            "oracle_zeros": int((o_ <= ZERO_TOL).sum()),
        }

    print("\nbirleşik havuza göre eşli farklar:")
    for name in CH:
        if name == base:
            continue
        for tag, dct in (("argmax-Q", sel), ("oracle", orc)):
            delta = dct[name] - dct[base]
            ci = boot(delta)
            p, pos, neg = sign_test(delta)
            print(f"  {name:30s} {tag:9s} {delta.mean():+8.4f}  "
                  f"[{ci[0]:+.4f},{ci[1]:+.4f}]  p={p:.4f}  ({pos} iyi / {neg} kötü)")
            summary[name][f"{tag}_delta_vs_combined"] = float(delta.mean())
            summary[name][f"{tag}_ci95"] = ci
            summary[name][f"{tag}_sign_p"] = float(p)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)
                                      + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
