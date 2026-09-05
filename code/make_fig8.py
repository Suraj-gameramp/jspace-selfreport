"""Figure 8: activation patching. Sufficiency and necessity of the answer-slot residual.

(a) Sufficiency: fraction of clean runs that name c after receiving the source's residual
    at (layer, position). Answer slot, summary token, assistant block. Secondary line: the
    median rank of c after the answer-slot patch.
(b) Necessity: fraction of named source runs that STOP naming c when the residual at
    (layer, position) is replaced by the clean run's value. Answer slot, user span.
(c) Per category at layer 28: how much does the name still depend on the user turn?
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..", "paper_data")
S = [json.loads(l) for l in open(os.path.join(D, "res_patch_sweep.jsonl"))]
N = [json.loads(l) for l in open(os.path.join(D, "res_necessity_sweep.jsonl"))]
LS = sorted(int(k) for k in S[0]["answer"])
LN = sorted(int(k) for k in N[0]["ablate_answer"])
CATS = ["countries", "animals", "foods", "colors", "emotions"]
CAT_COLOR = {"countries": "#0072B2", "animals": "#E69F00", "foods": "#009E73",
             "colors": "#CC79A7", "emotions": "#D55E00"}
RED, GREY, BLUE = "#B00020", "#888888", "#0072B2"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200,
    "pdf.fonttype": 42, "savefig.bbox": "tight",
})

named_S = [r for r in S if r["named_src"]]
named_N = [r for r in N if r["named_src"]]
suf = {k: [np.mean([r[k][str(l)] < 5 for r in named_S]) for l in LS] for k in ("answer", "summary", "block")}
suf_med = [np.median([r["answer"][str(l)] for r in named_S]) for l in LS]
nec = {k: [np.mean([r[k][str(l)] >= 5 for r in named_N]) for l in LN] for k in ("ablate_answer", "ablate_user_span")}

fig = plt.figure(figsize=(6.9, 5.0))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.85], hspace=0.55, wspace=0.55)
axA = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[0, 1]); axC = fig.add_subplot(gs[1, :])

# (a) sufficiency
axA.plot(LS, suf["answer"], "o-", color=RED, lw=1.6, ms=3.5, label="answer slot")
axA.plot(LS, suf["block"], "s--", color=RED, lw=1.1, ms=3, alpha=0.6, label="assistant span (block)")
axA.plot(LS, suf["summary"], "^:", color=GREY, lw=1.3, ms=3.5, label="summary token")
axA.axvspan(29, 34.5, color=RED, alpha=0.06, lw=0)
axA.text(31.7, 0.04, "circular", ha="center", fontsize=6.5, color=RED)
axA.set_xlim(11, 34.5); axA.set_ylim(-0.03, 1.03)
axA.set_xlabel("patch layer"); axA.set_ylabel("fraction of clean runs\nthat now name c")
axA.set_title("(a)  sufficiency: source $\\rightarrow$ clean", loc="left")
axA.legend(loc="center left", frameon=False, bbox_to_anchor=(0.0, 0.62))
axA.grid(color="#ededed", lw=0.5)
axA2 = axA.twinx()
axA2.plot(LS, suf_med, color="k", lw=0.9, ls="-.", alpha=0.7)
axA2.set_yscale("log"); axA2.set_ylim(1, 1e4)
axA2.set_ylabel("median rank of c\nafter answer-slot patch", fontsize=6.5, labelpad=2)
axA2.spines["right"].set_visible(True)
axA2.tick_params(labelsize=6.5)

# (b) necessity
axB.plot(LN, nec["ablate_answer"], "o-", color=RED, lw=1.6, ms=3.5, label="answer slot cleaned")
axB.plot(LN, nec["ablate_user_span"], "D-", color=BLUE, lw=1.6, ms=3.5, label="user span cleaned")
axB.axvline(23, color="k", lw=0.8, alpha=0.5)
axB.text(23.3, 0.55, "L23", fontsize=6.5)
axB.axvspan(29, 34.5, color=RED, alpha=0.06, lw=0)
axB.set_xlim(15, 32.5); axB.set_ylim(-0.03, 1.03)
axB.set_xlabel("ablation layer"); axB.set_ylabel("fraction of named runs\nthat stop naming c")
axB.set_title("(b)  necessity: clean $\\rightarrow$ source", loc="left")
axB.legend(loc="lower left", frameon=False, bbox_to_anchor=(0.0, 0.02))
axB.grid(color="#ededed", lw=0.5)

# (c) per category: user-span dependence at L28 vs sufficiency at L28
x = np.arange(len(CATS)); w = 0.36
uk = [np.mean([r["ablate_user_span"]["28"] >= 5 for r in named_N if r["category"] == c]) if any(r["category"] == c for r in named_N) else 0 for c in CATS]
sf = [np.mean([r["answer"]["28"] < 5 for r in named_S if r["category"] == c]) if any(r["category"] == c for r in named_S) else 0 for c in CATS]
nn = [sum(r["category"] == c for r in named_N) for c in CATS]
axC.bar(x - w / 2, uk, w, color=BLUE, label="name still needs the user turn at L28 (necessity)")
axC.bar(x + w / 2, sf, w, color=RED, alpha=0.85, label="answer-slot vector alone suffices at L28 (sufficiency)")
for i, c in enumerate(CATS):
    axC.text(i, -0.09, f"n={nn[i]}", ha="center", fontsize=6.5, color="#555")
axC.set_xticks(x); axC.set_xticklabels(CATS)
axC.set_ylim(-0.12, 1.05); axC.set_ylabel("fraction of named sources")
axC.set_title("(c)  at layer 28, how committed is the name to the answer slot?  (colors: no named sources)", loc="left")
axC.legend(loc="upper right", frameon=False)
axC.grid(axis="y", color="#ededed", lw=0.5)

for ext in ("pdf", "png"):
    fig.savefig(os.path.join(HERE, f"fig8.{ext}"))
plt.close(fig)

with open(os.path.join(HERE, "fig8_data.csv"), "w") as f:
    f.write("panel,series,layer_or_category,value\n")
    for k in suf:
        for l, v in zip(LS, suf[k]): f.write(f"a,{k},{l},{v:.4f}\n")
    for l, v in zip(LS, suf_med): f.write(f"a,median_rank_answer,{l},{v:.1f}\n")
    for k in nec:
        for l, v in zip(LN, nec[k]): f.write(f"b,{k},{l},{v:.4f}\n")
    for i, c in enumerate(CATS):
        f.write(f"c,user_span_kill_L28,{c},{uk[i]:.4f}\nc,answer_suff_L28,{c},{sf[i]:.4f}\n")

# recompute checks
assert abs(suf["answer"][LS.index(24)] - 0.0) < 1e-9
assert nec["ablate_answer"][LN.index(24)] == 1.0
assert nec["ablate_answer"][LN.index(23)] > 0.9
src = open(os.path.abspath(__file__)).read()
assert "\u2013" not in src and "\u2014" not in src
print(f"fig8 ok | named sources: suff {len(named_S)}, nec {len(named_N)}")
print(f"  sufficiency answer slot: L24 {suf['answer'][LS.index(24)]:.2f}  L28 {suf['answer'][LS.index(28)]:.2f}  L32 {suf['answer'][LS.index(32)]:.2f}")
print(f"  necessity answer slot:   L22 {nec['ablate_answer'][LN.index(22)]:.2f}  L23 {nec['ablate_answer'][LN.index(23)]:.2f}  L24 {nec['ablate_answer'][LN.index(24)]:.2f}")
print(f"  necessity user span:     L24 {nec['ablate_user_span'][LN.index(24)]:.2f}  L28 {nec['ablate_user_span'][LN.index(28)]:.2f}  L32 {nec['ablate_user_span'][LN.index(32)]:.2f}")
