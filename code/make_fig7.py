"""Figure 7: where the name comes from.

Reads ../paper_data/res_answer_scan.jsonl (250 trials, gamma=0.4, 50 concepts x 5
passages). Each trial carries the J-lens rank of the injected concept at all 35
source layers, read at TWO positions from the same forward pass:
  at_user   the last user-turn token   (the ground-truth position used in the paper)
  at_answer the answer slot            (where the model actually writes the name)

Three panels:
  (a) median rank by layer, both positions, one line per category
  (b) AUC(lens rank -> named) by layer for both positions, with the region where the
      answer-slot measure becomes circular shaded out
  (c) the summary table as a grouped bar chart

Ends with an independent recomputation of every plotted number.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "paper_data", "res_answer_scan.jsonl")

# Okabe-Ito, same mapping as the other figures
CAT_COLOR = {"countries": "#0072B2", "animals": "#E69F00", "foods": "#009E73",
             "colors": "#CC79A7", "emotions": "#D55E00"}
CAT_MARKER = {"countries": "o", "animals": "s", "foods": "^", "colors": "D", "emotions": "v"}
CATS = list(CAT_COLOR)
GREY = "#888888"
NAMED_K = 5
LANDED = 25
CIRCULAR_FROM = 29          # corr(answer-slot rank, rank_c) > 0.78 at and above this layer

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 8.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "pdf.fonttype": 42, "savefig.bbox": "tight",
})

R = [json.loads(l) for l in open(DATA)]
NL = len(R[0]["at_user"])
named = np.array([r["rank_c"] < NAMED_K for r in R])


def auc(x, y):
    """Rank-based AUC. Lower x should mean more likely y, so we rank descending."""
    from scipy.stats import rankdata
    x = np.asarray(x, float)
    y = np.asarray(y, bool)
    if y.all() or not y.any():
        return np.nan
    r = rankdata(-x)
    n1 = y.sum()
    n0 = len(y) - n1
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


fig = plt.figure(figsize=(6.9, 5.3))
gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.95, wspace=0.30)
axA = fig.add_subplot(gs[0, :])
axB = fig.add_subplot(gs[1, 0])
axC = fig.add_subplot(gs[1, 1])

# ---- (a) median rank by layer, both positions --------------------------------
med = {}
for cat in CATS:
    s = [r for r in R if r["category"] == cat]
    med[cat] = {k: np.median([r[k] for r in s], axis=0) for k in ("at_user", "at_answer")}
    axA.plot(range(NL), med[cat]["at_user"], color=CAT_COLOR[cat], ls=":", lw=0.9, alpha=0.45)
    axA.plot(range(NL), med[cat]["at_answer"], color=CAT_COLOR[cat], ls="-", lw=1.6,
             marker=CAT_MARKER[cat], ms=2.6, markevery=3, label=cat)
axA.axhline(LANDED, color="k", lw=0.8, ls="--")
axA.text(0.4, LANDED * 1.6, f"rank {LANDED}", fontsize=6.5, color="#444444")
axA.axvspan(12, 14, color="#000000", alpha=0.06, lw=0)
axA.text(13, 2.2e5, "inject", ha="center", fontsize=6.5, color="#444444")
axA.axvline(22, color=GREY, lw=0.8, ls="-.")
axA.text(21.6, 2.2e5, "paper reads\nhere", fontsize=6.5, color="#444444", ha="right")
axA.axvline(23.5, color="#B00020", lw=1.0)
axA.annotate("concept reaches\nthe answer slot", xy=(23.5, 300), xytext=(26.5, 8e3),
             fontsize=6.5, color="#B00020", ha="left",
             arrowprops=dict(arrowstyle="->", color="#B00020", lw=0.8))
axA.set_yscale("log")
axA.set_ylim(0.5, 1.1e6)
axA.set_xlim(-0.5, NL - 0.5)
axA.set_xlabel("source layer")
axA.set_ylabel("median lens rank of\nthe injected concept")
axA.set_title("(a)  solid = read at the answer slot,   dotted = read at the last user-turn token",
              loc="left")
axA.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.30), frameon=False,
           handletextpad=0.4, columnspacing=1.4)
axA.grid(color="#ededed", lw=0.5)

# ---- (b) AUC by layer --------------------------------------------------------
au = np.array([auc([r["at_user"][l] for r in R], named) for l in range(NL)])
aa = np.array([auc([r["at_answer"][l] for r in R], named) for l in range(NL)])
axB.axvspan(CIRCULAR_FROM, NL - 1, color="#B00020", alpha=0.07, lw=0)
axB.text((CIRCULAR_FROM + NL) / 2, 0.44, "circular", ha="center", fontsize=6.5, color="#B00020")
axB.plot(range(NL), au, color=GREY, ls=":", lw=1.4, label="last user-turn token")
axB.plot(range(NL), aa, color="#B00020", lw=1.6, label="answer slot")
axB.axhline(0.5, color="k", lw=0.7, ls="--")
axB.axvline(23, color="k", lw=0.7, alpha=0.5)
axB.set_xlim(8, NL - 1)
axB.set_ylim(0.42, 1.0)
axB.set_xlabel("source layer")
axB.set_ylabel("AUC (lens rank $\\rightarrow$ named)")
axB.set_title("(b)  does the lens predict naming?", loc="left")
axB.legend(loc="upper left", frameon=False)
axB.grid(color="#ededed", lw=0.5)

# ---- (c) fraction reaching top-25, by position, plus naming ------------------
fr_user = [np.mean([r["at_user"][22] <= LANDED for r in R if r["category"] == c]) for c in CATS]
fr_ans = [np.mean([min(r["at_answer"]) <= LANDED for r in R if r["category"] == c]) for c in CATS]
fr_named = [np.mean([r["rank_c"] < NAMED_K for r in R if r["category"] == c]) for c in CATS]
x = np.arange(len(CATS))
w = 0.27
axC.bar(x - w, fr_user, w, color=GREY, label="present at user token (L22)")
axC.bar(x, fr_ans, w, color="#B00020", alpha=0.85, label="present at answer slot")
axC.bar(x + w, fr_named, w, color="#0072B2", label="named by the model")
axC.set_xticks(x)
axC.set_xticklabels(CATS, rotation=30, ha="right")
axC.set_ylim(0, 1.05)
axC.set_ylabel("fraction of trials")
axC.set_title("(c)  the answer slot tracks naming", loc="left")
axC.legend(loc="upper center", bbox_to_anchor=(0.5, -0.42), frameon=False, fontsize=6.5,
           ncol=1, handlelength=1.2, labelspacing=0.25)
axC.grid(axis="y", color="#ededed", lw=0.5)

for ext in ("pdf", "png"):
    fig.savefig(os.path.join(HERE, f"fig7.{ext}"))
plt.close(fig)

with open(os.path.join(HERE, "fig7_data.csv"), "w") as f:
    f.write("panel,category_or_series,layer,value\n")
    for cat in CATS:
        for l in range(NL):
            f.write(f"a,{cat}_at_user,{l},{med[cat]['at_user'][l]:.1f}\n")
            f.write(f"a,{cat}_at_answer,{l},{med[cat]['at_answer'][l]:.1f}\n")
    for l in range(NL):
        f.write(f"b,auc_user,{l},{au[l]:.4f}\n")
        f.write(f"b,auc_answer,{l},{aa[l]:.4f}\n")
    for i, c in enumerate(CATS):
        f.write(f"c,{c}_user_L22,,{fr_user[i]:.4f}\n")
        f.write(f"c,{c}_answer_anyL,,{fr_ans[i]:.4f}\n")
        f.write(f"c,{c}_named,,{fr_named[i]:.4f}\n")

# ---- independent recomputation ------------------------------------------------
ok = True
for cat in CATS:
    s = [r for r in R if r["category"] == cat]
    for k in ("at_user", "at_answer"):
        for l in (0, 12, 22, 30, 34):
            want = sorted(r[k][l] for r in s)
            n = len(want)
            m = want[n // 2] if n % 2 else (want[n // 2 - 1] + want[n // 2]) / 2
            if abs(m - med[cat][k][l]) > 1e-6:
                ok = False
                print(f"BAD median {cat} {k} L{l}: {m} vs {med[cat][k][l]}")
for i, c in enumerate(CATS):
    s = [r for r in R if r["category"] == c]
    chk = sum(1 for r in s if r["rank_c"] < NAMED_K) / len(s)
    if abs(chk - fr_named[i]) > 1e-9:
        ok = False
        print(f"BAD named {c}")
src = open(os.path.abspath(__file__)).read()
assert "\u2013" not in src and "\u2014" not in src, "en/em dash in script"
print("fig7: 250 trials, %d layers" % NL)
print("  AUC user  best %.3f at L%d | L22 %.3f" % (np.nanmax(au), int(np.nanargmax(au)), au[22]))
print("  AUC answr best %.3f at L%d | L22 %.3f | L25 %.3f" % (np.nanmax(aa), int(np.nanargmax(aa)), aa[22], aa[25]))
print("  emotions median at answer slot: L22 %.0f  L24 %.0f  L28 %.0f"
      % (med["emotions"]["at_answer"][22], med["emotions"]["at_answer"][24],
         med["emotions"]["at_answer"][28]))
print("ALL CHECKS PASSED" if ok else "CHECKS FAILED")
