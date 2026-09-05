"""Analyse the natural-presence prompt filter.

Reads paper_data/res_natural_filter.jsonl. Per concept, two wordings (injection / content):
  lens_answer_L23, lens_answer_L24   lens rank of the target at the answer slot, no injection
  lens_user_L22                      lens rank at the old summary-token position
  rank_c                             rank of the target in the CONCEPT slot (does it name it unaided?)
  other_battery_L23                  any OTHER battery concept with lens rank <= 25 at L23 (leaks)

Filter rule (mentor): keep prompts whose concept is at lens rank <= 10 at the answer slot, L23.
"""
import json
import sys
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "paper_data/res_natural_filter.jsonl"
R = [json.loads(l) for l in open(path)]
CATS = ["countries", "animals", "foods", "colors", "emotions"]
KEEP = 10

for w in ("content", "injection"):
    rows = [r for r in R if r["wording"] == w]
    print(f"\n=== wording: {w} ({len(rows)} prompts) ===")
    keep = [r for r in rows if r["lens_answer_L23"] <= KEEP]
    print(f"pass filter (lens rank <= {KEEP} at answer slot L23): {len(keep)}/{len(rows)}")
    print(f"  median lens rank at answer L23: {np.median([r['lens_answer_L23'] for r in rows]):.0f}; "
          f"at user L22: {np.median([r['lens_user_L22'] for r in rows]):.0f}")
    print(f"  named unaided (rank_c < 5): {sum(r['rank_c'] < 5 for r in rows)}/{len(rows)}")
    print(f"\n  {'category':10s} {'pass':>5} {'med L23':>8} {'med L24':>8} {'med user22':>10} {'named':>6}")
    for cat in CATS:
        s = [r for r in rows if r["category"] == cat]
        print(f"  {cat:10s} {sum(r['lens_answer_L23'] <= KEEP for r in s):3d}/{len(s)} "
              f"{np.median([r['lens_answer_L23'] for r in s]):8.0f} {np.median([r['lens_answer_L24'] for r in s]):8.0f} "
              f"{np.median([r['lens_user_L22'] for r in s]):10.0f} {sum(r['rank_c'] < 5 for r in s):4d}/{len(s)}")

print("\n=== per concept, content wording: lens L23 at answer / rank_c / leaks ===")
rows = [r for r in R if r["wording"] == "content"]
for cat in CATS:
    print(f"  -- {cat}")
    for r in sorted([x for x in rows if x["category"] == cat], key=lambda x: x["lens_answer_L23"]):
        flag = "KEEP" if r["lens_answer_L23"] <= KEEP else "    "
        leaks = ", ".join(f"{k}:{v}" for k, v in sorted(r["other_battery_L23"].items(), key=lambda kv: kv[1])[:3])
        print(f"  {flag} {r['concept']:10s} L23={r['lens_answer_L23']:6d} L24={r['lens_answer_L24']:6d} slot={r['rank_c']:6d}  "
              f"top5slot={r['top5_slot']}  leaks={{{leaks}}}")

print("\n=== prompts that FAIL under content wording: what is at the top of the lens instead? ===")
for r in [x for x in rows if x["lens_answer_L23"] > KEEP]:
    print(f"  {r['concept']:10s} L23={r['lens_answer_L23']:6d}  lens top5={r['top5_lens23']}")
