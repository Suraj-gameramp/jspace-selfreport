"""Analyse the rank-one ablation pilot: content-necessity vs position-necessity.

Reads paper_data/res_rank_one_pilot.jsonl. Per (concept, passage) at gamma 0.4, per layer:
  full       answer-slot residual replaced by clean            (position necessity, reference)
  proj0      only v_c projected out, to zero                   (content necessity, strict)
  projclean  v_c projection set to the clean run's value       (content necessity, injection-only)
  projrand   a random J-space direction projected out          (null)
Each gives rank of c and log P(c) in the CONCEPT slot. Also proj_src / proj_clean: the scalar
projection of the source and clean residuals onto v_c.
"""
import json
import sys
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "paper_data/res_rank_one_pilot.jsonl"
R = [json.loads(l) for l in open(path)]
LAYERS = sorted(int(k) for k in R[0]["layers"])
EDITS = ["full", "proj0", "projclean", "projrand"]
CATS = ["countries", "animals", "foods", "colors", "emotions"]
named = [r for r in R if r["named_src"]]

print(f"{len(R)} trials; named sources {len(named)}; source median rank on named {np.median([r['rank_src'] for r in named]):.0f}, "
      f"median log P(c) {np.median([r['logp_src'] for r in named]):.2f}")

def kill(rows, l, e, k=5):
    return np.mean([r["layers"][str(l)][e]["rank"] >= k for r in rows]) if rows else np.nan
def dlogp(rows, l, e):
    return np.median([r["layers"][str(l)][e]["logp"] - r["logp_src"] for r in rows]) if rows else np.nan
def medrank(rows, l, e):
    return np.median([r["layers"][str(l)][e]["rank"] for r in rows]) if rows else np.nan

print("\nKILL RATE on named sources (fraction whose rank rises to >= 5), and median delta log P(c)")
print(f"{'layer':>5} | {'full':>14} | {'proj0':>14} | {'projclean':>14} | {'projrand':>14}")
print(f"{'':>5} | {'kill  dlogp':>14} | {'kill  dlogp':>14} | {'kill  dlogp':>14} | {'kill  dlogp':>14}")
for l in LAYERS:
    cells = [f"{kill(named,l,e):.2f} {dlogp(named,l,e):+6.2f}" for e in EDITS]
    print(f"{l:5d} | " + " | ".join(f"{c:>14}" for c in cells))

print("\nMEDIAN RANK after each edit, named sources")
print(f"{'layer':>5} | " + " | ".join(f"{e:>9}" for e in EDITS))
for l in LAYERS:
    print(f"{l:5d} | " + " | ".join(f"{medrank(named,l,e):9.0f}" for e in EDITS))

print("\nTHE QUESTION: does removing ONLY v_c kill naming about as often as removing everything?")
for l in (23, 24):
    f, p0, pc, pr = (kill(named, l, e) for e in EDITS)
    print(f"  L{l}: full {f:.2f}  proj0 {p0:.2f}  projclean {pc:.2f}  random-dir null {pr:.2f}"
          f"   -> proj0/full = {p0/f if f else float('nan'):.2f}")

print("\nHOW MUCH OF THE RESIDUAL IS ALONG v_c?  (projection onto unit v_c; source vs clean)")
for l in LAYERS:
    ps = np.median([r["layers"][str(l)]["proj_src"] for r in named]); pc = np.median([r["layers"][str(l)]["proj_clean"] for r in named])
    nrm = None
    print(f"  L{l}: source proj {ps:7.2f}   clean proj {pc:7.2f}   injection added {ps-pc:+7.2f}")

print("\nBY CATEGORY at L23, named sources: kill rate for full / proj0 / projclean")
for cat in CATS:
    s = [r for r in named if r["category"] == cat]
    if not s: print(f"  {cat:10s} (no named sources)"); continue
    print(f"  {cat:10s} n={len(s):3d}  full {kill(s,23,'full'):.2f}  proj0 {kill(s,23,'proj0'):.2f}  projclean {kill(s,23,'projclean'):.2f}  rand {kill(s,23,'projrand'):.2f}")

print("\nIMPORTANCE LABEL candidate: delta log P(c) under proj0 at L23, all 500 trials")
d = np.array([r["layers"]["23"]["proj0"]["logp"] - r["logp_src"] for r in R])
nm = np.array([r["named_src"] for r in R])
print(f"  named sources:     median {np.median(d[nm]):+.2f}  IQR [{np.percentile(d[nm],25):+.2f}, {np.percentile(d[nm],75):+.2f}]")
print(f"  not-named sources: median {np.median(d[~nm]):+.2f}  IQR [{np.percentile(d[~nm],25):+.2f}, {np.percentile(d[~nm],75):+.2f}]")
from scipy.stats import rankdata
def auc(x, y):
    x = np.asarray(x, float); y = np.asarray(y, bool); r = rankdata(x); n1 = y.sum(); n0 = len(y) - n1
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
print(f"  AUC(-dlogp separates named from not-named): {auc(-d, nm):.3f}")
