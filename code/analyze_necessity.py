"""Analyse the necessity (reverse-patch) sweep.

Reads paper_data/res_necessity_sweep.jsonl. Each row: one (concept, passage) at gamma 0.4 with
  rank_src            rank of c in the injected source run, no ablation
  named_src           rank_src < 5
  ablate_answer[l]    rank of c after overwriting the ANSWER-SLOT residual at lens-layer l
                      with the clean run's value, inside the injected run
  ablate_user_span[l] same, overwriting ALL injected user-turn positions at layer l

Two questions:
  1. Is the answer-slot residual at layer l NECESSARY for the name? (naming dies when it
     is replaced by clean)
  2. Is the injected user span still being read at layer l? (naming dies when the user
     positions are cleaned at l, even though the answer slot keeps its injected history)
"""
import sys
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "paper_data/res_necessity_sweep.jsonl"
import json
R = [json.loads(l) for l in open(path)]
LAYERS = sorted(int(k) for k in R[0]["ablate_answer"])
CATS = ["countries", "animals", "foods", "colors", "emotions"]
named = [r for r in R if r["named_src"]]

print(f"{len(R)} trials; source named c on {len(named)}")
print(f"source median rank_c on named trials: {np.median([r['rank_src'] for r in named]):.0f}")

def kill_rate(rows, key, l, k=5):
    """fraction of named sources whose naming is destroyed (rank >= k) by the ablation"""
    return np.mean([r[key][str(l)] >= k for r in rows]) if rows else np.nan

def med(rows, key, l):
    return np.median([r[key][str(l)] for r in rows]) if rows else np.nan

print("\nKILL RATE = fraction of named sources that STOP naming c after the ablation")
print(f"{'layer':>6} | {'answer slot':>11} {'med rank':>9} | {'user span':>9} {'med rank':>9}")
for l in LAYERS:
    print(f"{l:6d} | {kill_rate(named,'ablate_answer',l):11.2f} {med(named,'ablate_answer',l):9.0f} | "
          f"{kill_rate(named,'ablate_user_span',l):9.2f} {med(named,'ablate_user_span',l):9.0f}")

print("\nREADING THE TWO COLUMNS TOGETHER")
print("  answer-slot kill high at layer l  -> the answer-slot residual at l is NECESSARY for the name")
print("  user-span kill high at layer l    -> late layers are still READING the injection from the user turn at l")
print("  both low at l                     -> the name is already committed downstream of l, or is redundant")

print("\nEARLIEST layer where cleaning the answer slot kills naming on >= 50% / >= 80% of named sources:")
for thr in (0.5, 0.8):
    hits = [l for l in LAYERS if kill_rate(named, "ablate_answer", l) >= thr]
    print(f"  >= {thr:.0%}: {'layer ' + str(min(hits)) if hits else 'never'}")
print("LATEST layer where cleaning the user span still kills naming on >= 50% of named sources:")
hits = [l for l in LAYERS if kill_rate(named, "ablate_user_span", l) >= 0.5]
print(f"  {'layer ' + str(max(hits)) if hits else 'never'}")

print("\nBY CATEGORY, named sources only: answer-slot kill at L24 / L28 ; user-span kill at L24 / L28")
print(f"{'category':10s} {'n':>4} | {'ans L24':>7} {'ans L28':>7} | {'usr L24':>7} {'usr L28':>7}")
for cat in CATS:
    s = [r for r in named if r["category"] == cat]
    if not s:
        print(f"{cat:10s} {0:4d} | (no named sources)"); continue
    print(f"{cat:10s} {len(s):4d} | {kill_rate(s,'ablate_answer',24):7.2f} {kill_rate(s,'ablate_answer',28):7.2f} | "
          f"{kill_rate(s,'ablate_user_span',24):7.2f} {kill_rate(s,'ablate_user_span',28):7.2f}")
