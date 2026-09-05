"""Analyse the scaled natural pool (natural_filter_v2) and, if present, the importance pilot.

natural_pool_labels.jsonl, per (concept, description):
  lens_L24, rank_c, p_present, mass, proj_clean, label in {present, middle, absent}, leaks, top5_slot
importance_natural.jsonl, per natural PRESENT prompt:
  rank_base, logp_base, named_base, projclean{rank,logp,dlogp}, proj0{...}, clean_proj_used, proj_natural
"""
import json
import os
import sys
import numpy as np
from scipy.stats import rankdata, spearmanr

D = "paper_data"
CATS = ["countries", "animals", "foods", "colors", "emotions"]

def auc(x, y):
    x = np.asarray(x, float); y = np.asarray(y, bool)
    if y.all() or not y.any(): return np.nan
    r = rankdata(-x); n1 = y.sum(); n0 = len(y) - n1
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

# ---------------------------------------------------------------- natural pool
p = os.path.join(D, "res_natural_pool_labels.jsonl")
R = [json.loads(l) for l in open(p)]
print(f"NATURAL POOL: {len(R)} descriptions, {len(set(r['concept'] for r in R))} concepts")
lab = lambda s, k: sum(r["label"] == k for r in s)
print(f"  present {lab(R,'present')}   middle {lab(R,'middle')}   absent {lab(R,'absent')}   (target: 150 to 250 present)")
print(f"  named unaided {sum(r['rank_c']<5 for r in R)}/{len(R)}   median lens {np.median([r['lens_L24'] for r in R]):.0f}   mass<0.5 {sum(r['mass']<0.5 for r in R)}")
print(f"\n  {'category':10s} {'n':>3} {'present':>8} {'middle':>7} {'absent':>7} | {'named':>6} {'med lens':>9} {'AUC lens->named':>15}")
for c in CATS:
    s = [r for r in R if r["category"] == c]
    print(f"  {c:10s} {len(s):3d} {lab(s,'present'):8d} {lab(s,'middle'):7d} {lab(s,'absent'):7d} | "
          f"{np.mean([r['rank_c']<5 for r in s]):6.2f} {np.median([r['lens_L24'] for r in s]):9.0f} "
          f"{auc([r['lens_L24'] for r in s],[r['rank_c']<5 for r in s]):15.3f}")
print("\n  per-concept present count (need >= 3 for a usable training stratum after the split):")
byc = {}
for r in R: byc.setdefault(r["concept"], []).append(r)
thin = [(c, lab(v, "present")) for c, v in byc.items() if lab(v, "present") < 3]
print("   concepts with < 3 present:", thin if thin else "none")
leaky = [(r["concept"], r["desc_idx"], r["leaks"]) for r in R if r["leaks"]]
print(f"\n  descriptions with another battery concept at lens rank <= 25: {len(leaky)}")
for c, i, lk in leaky[:12]: print(f"    {c:10s} #{i}: {lk}")
print(f"\n  covariate: proj_clean median {np.median([r['proj_clean'] for r in R]):+.2f}; "
      f"Spearman(proj_clean, lens_L24) {spearmanr([r['proj_clean'] for r in R],[r['lens_L24'] for r in R])[0]:+.3f}; "
      f"Spearman(proj_clean, rank_c) {spearmanr([r['proj_clean'] for r in R],[r['rank_c'] for r in R])[0]:+.3f}")
print("\n  dose response within the natural stratum (pre-registered secondary prediction, baseline):")
for lo, hi in ((0, 10), (11, 100), (101, 999), (1000, 10**9)):
    s = [r for r in R if lo <= r["lens_L24"] <= hi]
    if s: print(f"    lens {lo:>4}-{hi if hi<10**9 else 'inf':<5}: n={len(s):3d}  named {np.mean([r['rank_c']<5 for r in s]):.2f}  P(present) {np.mean([r['p_present'] for r in s]):.2f}")

# ---------------------------------------------------------------- importance pilot
q = os.path.join(D, "res_importance_natural.jsonl")
if not os.path.exists(q):
    print("\nIMPORTANCE PILOT: not yet run"); sys.exit()
I = [json.loads(l) for l in open(q)]
nm = np.array([r["named_base"] for r in I])
print(f"\nIMPORTANCE PILOT (natural present prompts): {len(I)} prompts, named at baseline {nm.mean():.2f}")
for k in ("projclean", "proj0"):
    d = np.array([r[k]["dlogp"] for r in I]); dr = np.array([np.log10(r[k]["rank"]+1) - np.log10(r["rank_base"]+1) for r in I])
    kill = np.mean([r[k]["rank"] >= 5 for r in I if r["named_base"]]) if nm.any() else np.nan
    print(f"  {k:9s} dlogp median {np.median(d):+.2f}  sd {d.std():.2f}  IQR [{np.percentile(d,25):+.2f},{np.percentile(d,75):+.2f}] | "
          f"AUC(-dlogp -> named) {auc(-d, nm):.3f}   AUC(drank -> named) {auc(dr, nm):.3f}   kill rate on named {kill:.2f}")
print("  verdict rule: rung survives only if sd is non-trivial AND AUC is well above 0.5 (threshold to be set on a held-out slice).")
print("  by category, projclean:")
for c in CATS:
    s = [r for r in I if r["category"] == c]
    if not s: continue
    d = np.array([r["projclean"]["dlogp"] for r in s]); y = np.array([r["named_base"] for r in s])
    print(f"    {c:10s} n={len(s):3d}  dlogp median {np.median(d):+.2f} sd {d.std():.2f}  AUC {auc(-d, y):.3f}")
