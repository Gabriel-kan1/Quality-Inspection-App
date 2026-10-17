"""Results figures from the 40-photo study.

argv: <scratch dir>
"""
import json
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

S = sys.argv[1]
summ = json.load(open(f"{S}/results/testset_summary.json"))
exp = json.load(open(f"{S}/results/testset_export.json"))
METHODS = [f"v{i}" for i in range(1, 19)]
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": GREY,
                     "xtick.color": GREY, "ytick.color": GREY})

# ── detection: features found against extra boxes per photograph ──────────────────────────
det = summ["detection"]
fig, ax = plt.subplots(figsize=(6.5, 3.3))
for m in METHODS:
    d = det[m]
    x, y = d["extras"] / d["photos"], d["recall"] * 100
    ax.scatter(x + 0.05, y, s=34, color=BLUE if int(m[1:]) <= 10 else ORANGE, zorder=3,
               edgecolor="white", linewidth=0.8)
    off = {"v12": (-24, -12), "v16": (5, -12), "v3": (-16, 4), "v5": (5, 4), "v18": (5, -10)}.get(m, (5, 3))
    ax.annotate(m, (x + 0.05, y), textcoords="offset points", xytext=off, fontsize=9, color="#0b0b0b")
ax.set_xscale("log")
ax.set_xlabel("Extra boxes per photograph (log scale)")
ax.set_ylabel("Reference features found (%)")
ax.grid(color="#e6e5e0", lw=0.6, zorder=0)
ax.scatter([], [], color=BLUE, label="Hybrid v1 to v10")
ax.scatter([], [], color=ORANGE, label="Hybrid v11 to v18")
ax.legend(frameon=False, fontsize=10, loc="lower left")
fig.tight_layout(); fig.savefig(f"{S}/figs/detection_tradeoff.png", dpi=300)

# ── classification: DINOv2 against CLIP on found features ─────────────────────────────────
cls = summ["classification"]
fig, ax = plt.subplots(figsize=(6.5, 2.9))
x = np.arange(len(METHODS))
dv = [cls["dino"][m]["acc"] * 100 for m in METHODS]
cv = [cls["clip"][m]["acc"] * 100 for m in METHODS]
for i in range(len(METHODS)):
    ax.plot([i, i], [cv[i], dv[i]], color="#c9c8c2", lw=1.5, zorder=1)
ax.scatter(x, dv, s=36, color=BLUE, label="DINOv2-S", zorder=3)
ax.scatter(x, cv, s=36, color=ORANGE, label="CLIP ViT-B/32", zorder=3)
ax.set_xticks(x, [m[1:] for m in METHODS], fontsize=10)
ax.set_xlabel("Hybrid edge method")
ax.set_ylabel("Found features\ncorrectly classified (%)")
ax.set_ylim(55, 90)
ax.grid(axis="y", color="#e6e5e0", lw=0.6, zorder=0)
ax.legend(frameon=False, fontsize=10, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.0))
fig.tight_layout(); fig.savefig(f"{S}/figs/encoder_testset.png", dpi=300)

# ── alignment gate: same-region against other-region similarity ───────────────────────────
gate = exp["gate"]
refs = {"4 screws": "4 screws normal", "3 thread holes": "3 thread holes normal", "3 holes": "3 holes normal"}
def region_of(n):
    return next(r for r in ("3 thread holes", "3 holes", "4 screws") if n.startswith(r + " "))
same, diff = [], []
for r, ref in refs.items():
    for n, s in gate[ref].items():
        if n != ref:
            (same if region_of(n) == r else diff).append(s)
fig, ax = plt.subplots(figsize=(6.5, 2.6))
bins = np.linspace(0.2, 1.0, 33)
ax.hist(diff, bins=bins, color=ORANGE, alpha=0.6, label=f"Other region (n = {len(diff)})")
ax.hist(same, bins=bins, color=BLUE, alpha=0.6, label=f"Same region (n = {len(same)})")
ax.axvline(0.70, color="#b85450", ls="--", lw=1.5)
ax.text(0.705, ax.get_ylim()[1] * 0.92, "threshold 0.70", color="#b85450", fontsize=10)
ax.set_xlabel("Cosine similarity between photograph and region reference")
ax.set_ylabel("Photographs")
ax.legend(frameon=False, fontsize=10, loc="upper left")
ax.grid(axis="y", color="#e6e5e0", lw=0.6); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(f"{S}/figs/gate_testset.png", dpi=300)

# ── confusion matrices: selector with counts only, and with the positional check ──────────
ver = summ["verdicts"]["dino"]
best_fixed = max(METHODS, key=lambda m: (ver[m]["count"]["tn"] + ver[m]["count"]["tp"], ver[m]["count"]["tp"]))
cases = [(f"a) Hybrid {best_fixed[1:]}, counts only".replace("Hybrid ", "Hybrid v"), ver[best_fixed]["count"]),
         ("b) Selector, counts only", ver["adaptive"]["count"]),
         ("c) Selector + positional", ver["adaptive"]["full"])]
fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.6))
ramp = matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#f4f8fd", BLUE])
for ax, (title, c) in zip(axes, cases):
    M = np.array([[c["tp"], c["fn"]], [c["fp"], c["tn"]]])
    share = np.array([[c["tp"] / 3, c["fn"] / 3], [c["fp"] / 27, c["tn"] / 27]])
    ax.imshow(share, cmap=ramp, vmin=0, vmax=1)
    for (i, j), v in np.ndenumerate(M):
        lab = [["TP", "FN"], ["FP", "TN"]][i][j]
        ax.text(j, i, f"{lab}\n{v}", ha="center", va="center", fontsize=11,
                color="white" if share[i, j] > 0.6 else "#0b0b0b")
    ax.set_xticks([0, 1], ["FAIL", "PASS"]); ax.set_yticks([0, 1], ["Defective", "Correct"])
    ax.set_xlabel("Verdict"); ax.set_title(title, fontsize=10.5); ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
axes[0].set_ylabel("Actual part")
for ax in axes[1:]:
    ax.set_yticklabels([])
fig.tight_layout(); fig.savefig(f"{S}/figs/confusion_selector.png", dpi=300)
print("best fixed (counts):", best_fixed, ver[best_fixed]["count"])
