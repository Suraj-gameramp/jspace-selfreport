"""Analyse the natural-prompt layer sweep: is there a layer at the answer slot, below the
circular band, where naturally implied concepts are readable?

Reads paper_data/res_natural_layer_sweep.jsonl. Per (concept, description):
  rank_c              rank of c in the CONCEPT slot (unaided naming)
  jlens[l], logitlens[l]   lens rank of c at the answer slot, layers 20..32
"""
import json
import sys
import numpy as np
from scipy.stats import rankdata

path = sys.argv[1] if len(sys.argv) > 1 else "paper_data/res_natural_layer_sweep.jsonl"
R = [json.loads(l) for l in open(path)]
LAYERS = sorted(int(k) for k in R[0]["jlens"])
CATS = ["countries", "animals", "foods", "colors", "emotions"]
named = np.array([r["rank_c"] < 5 for r in R])

def auc(x, y):
    x = np.asarray(x, float); y = np.asarray(y, bool)
    if y.all() or not y.any(): return np.nan
    r = rankdata(-x); n1 = y.sum(); n0 = len(y) - n1
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

print(f"{len(R)} natural prompts; named unaided {named.sum()} ({named.mean():.2f})")
print("\nPER LAYER, J-lens at the answer slot: how many prompts pass each threshold, and how well the rank predicts unaided naming")
print(f"{'L':>3} | {'<=10':>5} {'<=100':>6} {'<=1000':>7} | {'median':>7} | {'AUC->named':>10} | {'named & >=1000':>14} | {'logit-lens <=100':>16} {'logit AUC':>9}")
for l in LAYERS:
    j = np.array([r["jlens"][str(l)] for r in R]); g = np.array([r["logitlens"][str(l)] for r in R])
    print(f"{l:3d} | {int((j<=10).sum()):5d} {int((j<=100).sum()):6d} {int((j<=1000).sum()):7d} | {np.median(j):7.0f} | {auc(j,named):10.3f} | "
          f"{int(((j>=1000)&named).sum()):14d} | {int((g<=100).sum()):16d} {auc(g,named):9.3f}")
print("\n  'named & >=1000' = prompts the model names unaided that the label would call ABSENT at that layer (label noise in the absent class).")
print("  Circular band starts at L29 (corr with output > 0.78 in the injected data); read L24 to L28 as the candidates.")

print("\nBY CATEGORY: fraction <= 100 at each candidate layer (J-lens)")
print(f"{'category':10s} " + " ".join(f"{'L'+str(l):>6}" for l in (22, 24, 26, 28, 30)) + "   named")
for c in CATS:
    s = [r for r in R if r["category"] == c]
    print(f"{c:10s} " + " ".join(f"{np.mean([r['jlens'][str(l)]<=100 for r in s]):6.2f}" for l in (22, 24, 26, 28, 30)) +
          f"   {np.mean([r['rank_c']<5 for r in s]):.2f}")

print("\nCONCEPTS WITH >= 3 descriptions at <= 100, by layer (the training-usability count)")
for l in (24, 26, 28):
    cnt = {}
    for r in R: cnt[r["concept"]] = cnt.get(r["concept"], 0) + (r["jlens"][str(l)] <= 100)
    ok = sum(v >= 3 for v in cnt.values())
    print(f"  L{l}: {ok}/50 concepts have >= 3 present descriptions; total present {sum(cnt.values())}")
