"""Analyse the activation-patching sweep.

Reads paper_data/res_patch_sweep.jsonl. Each row: one (concept, passage) at gamma 0.4 with
  rank_src     rank of c in the injected source run's CONCEPT slot
  rank_clean   rank of c in the clean target run (no injection, no patch)
  named_src    rank_src < 5
  answer[l]    rank of c after patching the ANSWER-SLOT residual at lens-layer l into clean
  summary[l]   same, patching the SUMMARY TOKEN (last user-turn token)
  block[l]     same, patching the whole assistant span as a block
  ctrl_clean_to_clean[l]   clean->clean patch at the answer slot (must do nothing)

The question: at which (layer, position) does the patch make the clean run name c?
"""
import json
import sys

import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "paper_data/res_patch_sweep.jsonl"
R = [json.loads(l) for l in open(path)]
LAYERS = sorted(int(k) for k in R[0]["answer"])
CATS = ["countries", "animals", "foods", "colors", "emotions"]
named_src = [r for r in R if r["named_src"]]
not_named = [r for r in R if not r["named_src"]]

print(f"{len(R)} trials; source named c on {len(named_src)}, not on {len(not_named)}")
print(f"clean baseline: median rank_c {np.median([r['rank_clean'] for r in R]):.0f}, "
      f"named on {sum(r['rank_clean'] < 5 for r in R)}/{len(R)}")

# ---- control: clean -> clean must do nothing -------------------------------------
c = [r["ctrl_clean_to_clean"]["24"] for r in R if "24" in r["ctrl_clean_to_clean"]]
print(f"\nCONTROL clean->clean patch at L24 answer slot: median rank {np.median(c):.0f}, "
      f"named {sum(x < 5 for x in c)}/{len(c)}   (should match clean baseline)")

# ---- main result: transfer rate by layer and position ------------------------------
def transfer(rows, key, l, k=5):
    return np.mean([r[key][str(l)] < k for r in rows]) if rows else np.nan

print("\nTRANSFER RATE = fraction of patched clean runs that name c (rank < 5)")
print("Sources that DID name c (the sufficiency test):")
print(f"{'layer':>6} {'answer slot':>12} {'summary tok':>12} {'assist block':>13}   median rank after answer-slot patch")
for l in LAYERS:
    med = np.median([r["answer"][str(l)] for r in named_src])
    print(f"{l:6d} {transfer(named_src,'answer',l):12.2f} {transfer(named_src,'summary',l):12.2f} "
          f"{transfer(named_src,'block',l):13.2f}   {med:8.0f}")

print("\nSources that did NOT name c (does partial information transfer?):")
print(f"{'layer':>6} {'answer slot':>12} {'summary tok':>12} {'assist block':>13}")
for l in LAYERS:
    print(f"{l:6d} {transfer(not_named,'answer',l):12.2f} {transfer(not_named,'summary',l):12.2f} "
          f"{transfer(not_named,'block',l):13.2f}")

# ---- earliest sufficient layer at the answer slot ------------------------------------
print("\nEARLIEST LAYER where the answer-slot patch transfers on >= 50% / >= 80% of named sources:")
for thr in (0.5, 0.8):
    hits = [l for l in LAYERS if transfer(named_src, "answer", l) >= thr]
    print(f"  >= {thr:.0%}: {'layer ' + str(min(hits)) if hits else 'never'}")

# ---- does the patch reproduce the source's rank, or just push it below 5? -------------
print("\nHow faithfully does the answer-slot patch reproduce the source rank? (Spearman, named+not)")
from scipy.stats import spearmanr
for l in (20, 22, 23, 24, 26, 28, 32, 34):
    rho = spearmanr([r["answer"][str(l)] for r in R], [r["rank_src"] for r in R])[0]
    print(f"  L{l:2d}: rho = {rho:+.3f}")

# ---- by category ------------------------------------------------------------------------
print("\nBY CATEGORY, answer-slot patch, named sources only: transfer at L22 / L24 / L28")
print(f"{'category':10s} {'n named':>8} {'L22':>6} {'L24':>6} {'L28':>6}   {'summary L24':>11} {'block L24':>9}")
for cat in CATS:
    s = [r for r in named_src if r["category"] == cat]
    if not s:
        print(f"{cat:10s} {0:8d}   (no named sources)"); continue
    print(f"{cat:10s} {len(s):8d} {transfer(s,'answer',22):6.2f} {transfer(s,'answer',24):6.2f} "
          f"{transfer(s,'answer',28):6.2f}   {transfer(s,'summary',24):11.2f} {transfer(s,'block',24):9.2f}")

# ---- the emotions/colors check ---------------------------------------------------------
print("\nCOLORS (never named in Phase 1): can a patch make the model name a color at all?")
co = [r for r in R if r["category"] == "colors"]
for l in (24, 28, 32, 34):
    print(f"  L{l}: answer-slot patch names a color on {sum(r['answer'][str(l)] < 5 for r in co)}/{len(co)}; "
          f"median rank {np.median([r['answer'][str(l)] for r in co]):.0f}")
