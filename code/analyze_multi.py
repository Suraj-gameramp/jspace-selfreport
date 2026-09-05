"""Two analyses for the option-(c) follow-ups.

A. labels_neutral_multi.jsonl (injected stratum at L24, L25, L26): does the injected stratum
   change if it adopts the same two-layer L25+L26 rule as the natural stratum? Counts under
   the injected margins (present <= 10, absent >= 500) at each single layer and under
   two-layer agreement, plus AUC(lens -> named) at each.
B. importance_2layer.jsonl (H5 validation on the two-layer natural present set): delta log P(c)
   under clean-value rank-one ablation at L25 and at L26, variance, AUC against unaided naming,
   agreement between the two layers.
"""
import json
import os
import numpy as np
from scipy.stats import rankdata, spearmanr

D = "paper_data"
CATS = ["countries", "animals", "foods", "colors", "emotions"]
INJ_P, INJ_A = 10, 500

def auc(x, y):
    x = np.asarray(x, float); y = np.asarray(y, bool)
    if y.all() or not y.any(): return np.nan
    r = rankdata(-x); n1 = y.sum(); n0 = len(y) - n1
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

# ---------------------------------------------------------------- A
p = os.path.join(D, "res_labels_neutral_multi.jsonl")
if os.path.exists(p):
    R = [json.loads(l) for l in open(p)]
    print(f"A. INJECTED STRATUM AT THREE LAYERS: {len(R)} rows")
    for g in (0.0, 0.2, 0.4):
        s = [r for r in R if r["gamma"] == g]; nm = np.array([r["rank_c"] < 5 for r in s])
        print(f"\n  gamma {g}: naming {nm.mean():.3f}")
        print(f"  {'scheme':>12} {'present':>8} {'middle':>7} {'absent':>7} {'AUC->named':>10} {'named&absent':>12}")
        for l in ("24", "25", "26"):
            v = np.array([r["lens"][l] for r in s])
            pres, absn = v <= INJ_P, v >= INJ_A
            print(f"  {'L'+l:>12} {int(pres.sum()):8d} {int((~pres & ~absn).sum()):7d} {int(absn.sum()):7d} {auc(v, nm):10.3f} {int((absn & nm).sum()):12d}")
        a = np.array([r["lens"]["25"] for r in s]); b = np.array([r["lens"]["26"] for r in s])
        pres2 = (a <= INJ_P) & (b <= INJ_P); abs2 = (a >= INJ_A) & (b >= INJ_A); disc = ~pres2 & ~abs2
        print(f"  {'L25+L26':>12} {int(pres2.sum()):8d} {int(disc.sum()):7d} {int(abs2.sum()):7d} {'':>10} {int((abs2 & nm).sum()):12d}   (middle column = discarded)")
    print("\n  circularity on injected trials, corr(log lens, log rank_c) at gamma 0.4:")
    s = [r for r in R if r["gamma"] == 0.4]; rc = np.log10(np.array([r["rank_c"] for r in s]) + 1)
    for l in ("24", "25", "26"):
        print(f"    L{l}: {np.corrcoef(np.log10(np.array([r['lens'][l] for r in s]) + 1), rc)[0,1]:.3f}")
    print("\n  by category at gamma 0.4, two-layer present rate vs L24 present rate:")
    for c in CATS:
        t = [r for r in s if r["category"] == c]
        a = np.array([r["lens"]["25"] for r in t]); b = np.array([r["lens"]["26"] for r in t]); v = np.array([r["lens"]["24"] for r in t])
        print(f"    {c:10s} L24 {np.mean(v <= INJ_P):.2f}   L25+26 {np.mean((a <= INJ_P) & (b <= INJ_P)):.2f}   discarded {np.mean(~((a<=INJ_P)&(b<=INJ_P)) & ~((a>=INJ_A)&(b>=INJ_A))):.2f}")
else:
    print("A. labels_neutral_multi not yet available")

# ---------------------------------------------------------------- B
q = os.path.join(D, "res_importance_2layer.jsonl")
if os.path.exists(q):
    I = [json.loads(l) for l in open(q)]
    nm = np.array([r["named_base"] for r in I])
    print(f"\n\nB. IMPORTANCE PILOT, TWO-LAYER NATURAL PRESENT SET: {len(I)} prompts, named at baseline {nm.mean():.2f}")
    for l in ("L25", "L26"):
        d = np.array([r[l]["dlogp"] for r in I]); kill = np.mean([r[l]["rank"] >= 5 for r in I if r["named_base"]]) if nm.any() else np.nan
        print(f"  {l}: dlogp median {np.median(d):+.2f}  sd {d.std():.2f}  IQR [{np.percentile(d,25):+.2f}, {np.percentile(d,75):+.2f}]  "
              f"AUC(-dlogp -> named) {auc(-d, nm):.3f}  kill rate on named {kill:.2f}")
    d25 = np.array([r["L25"]["dlogp"] for r in I]); d26 = np.array([r["L26"]["dlogp"] for r in I])
    print(f"  agreement between layers: Spearman(dlogp L25, dlogp L26) = {spearmanr(d25, d26)[0]:.3f}")
    print(f"  by category (L26): " + "; ".join(
        f"{c} n={sum(r['category']==c for r in I)} med {np.median([r['L26']['dlogp'] for r in I if r['category']==c]):+.2f}"
        for c in CATS if any(r["category"] == c for r in I)))
    print("  verdict: rung live iff sd non-trivial AND AUC well above 0.5 at the chosen layer; threshold set on a held-out slice.")
else:
    print("\nB. importance_2layer not yet available")
