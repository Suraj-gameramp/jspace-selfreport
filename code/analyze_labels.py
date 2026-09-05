"""Analyse the injected-stratum labels regenerated under the neutral elicitation.

Reads paper_data/res_labels_neutral.jsonl. Per (concept, passage, gamma in {0, 0.2, 0.4}):
  lens_L24    lens rank of c at the answer slot, hidden_states[25]   <- the label
  rank_c      rank of c in the CONCEPT slot                          <- naming
  p_present   forced-choice yes after 'PRESENT:'
  proj_src, proj_clean   <h, v_c> at L24 answer slot for the injected and clean residual

Applies the frozen injected-stratum margins: present <= 10, absent >= 500, middle discarded.
"""
import json
import sys
import numpy as np
from scipy.stats import rankdata, spearmanr

path = sys.argv[1] if len(sys.argv) > 1 else "paper_data/res_labels_neutral.jsonl"
R = [json.loads(l) for l in open(path)]
CATS = ["countries", "animals", "foods", "colors", "emotions"]
PRES, ABS = 10, 500

def auc(x, y):
    x = np.asarray(x, float); y = np.asarray(y, bool)
    if y.all() or not y.any(): return np.nan
    r = rankdata(-x); n1 = y.sum(); n0 = len(y) - n1
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

print(f"{len(R)} rows")
print("\nLABEL COUNTS under frozen margins (present <= 10, absent >= 500), by gamma")
print(f"{'gamma':>5} {'n':>5} {'present':>8} {'middle':>7} {'absent':>7} | {'named':>6} {'P(present)':>10} {'mass<0.5':>8}")
for g in (0.0, 0.2, 0.4):
    s = [r for r in R if r["gamma"] == g]
    v = np.array([r["lens_L24"] for r in s])
    print(f"{g:5.1f} {len(s):5d} {int((v<=PRES).sum()):8d} {int(((v>PRES)&(v<ABS)).sum()):7d} {int((v>=ABS).sum()):7d} | "
          f"{np.mean([r['rank_c']<5 for r in s]):6.3f} {np.mean([r['p_present'] for r in s]):10.3f} {np.mean([r['mass']<0.5 for r in s]):8.3f}")

print("\nBY CATEGORY at gamma 0.4: present rate, named rate, and AUC(lens_L24 -> named) under the neutral wording")
for cat in CATS:
    s = [r for r in R if r["gamma"] == 0.4 and r["category"] == cat]
    v = np.array([r["lens_L24"] for r in s]); nm = np.array([r["rank_c"] < 5 for r in s])
    print(f"  {cat:10s} present {np.mean(v<=PRES):.2f}  middle {np.mean((v>PRES)&(v<ABS)):.2f}  absent {np.mean(v>=ABS):.2f} | "
          f"named {nm.mean():.2f}  AUC {auc(v, nm):.3f}  median lens {np.median(v):.0f}")

print("\nCOMPARISON with the Phase 1 injection wording (gamma 0.4, all trials): pooled AUC and naming")
s = [r for r in R if r["gamma"] == 0.4]
v = np.array([r["lens_L24"] for r in s]); nm = np.array([r["rank_c"] < 5 for r in s])
print(f"  neutral wording, L24 answer slot: AUC {auc(v, nm):.3f}   named {nm.mean():.3f}   present {np.mean(v<=PRES):.3f}")
print(f"  (Phase 1, injection wording, user token L22: AUC 0.556, named 0.216, landed 0.307)")

print("\nTHE COVARIATE: clean projection <h_clean, v_c> at L24 answer slot")
pc = np.array([r["proj_clean"] for r in R if r["gamma"] == 0.0])
print(f"  median {np.median(pc):+.2f}  IQR [{np.percentile(pc,25):+.2f}, {np.percentile(pc,75):+.2f}]  fraction negative {np.mean(pc<0):.2f}")
for cat in CATS:
    x = [r["proj_clean"] for r in R if r["gamma"] == 0.0 and r["category"] == cat]
    print(f"    {cat:10s} median {np.median(x):+.2f}")
s4 = [r for r in R if r["gamma"] == 0.4]
print(f"  does clean projection predict naming at gamma 0.4?  Spearman(proj_clean, rank_c) = {spearmanr([r['proj_clean'] for r in s4],[r['rank_c'] for r in s4])[0]:+.3f}")
print(f"  does it predict landing?  Spearman(proj_clean, lens_L24) = {spearmanr([r['proj_clean'] for r in s4],[r['lens_L24'] for r in s4])[0]:+.3f}")

print("\nDOSE RESPONSE of the label itself (median lens_L24 by gamma and category)")
for cat in CATS:
    print(f"  {cat:10s} " + "  ".join(f"g{g}: {np.median([r['lens_L24'] for r in R if r['gamma']==g and r['category']==cat]):7.0f}" for g in (0.0, 0.2, 0.4)))
