"""Recall-QPS curves of DiskANN and MCGI on SIFT1M, GloVe-100, GIST1M, and SIFT1B (figures/recall_qps.pdf).

Data: data/recall_qps_clean.csv, exported from the clean serial runs (one search per method, page cache
dropped, idle machine). Both methods search the same index; MCGI adapts the search list per query.
"""
import csv
import os

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "figures", "recall_qps.pdf")
PANELS = [("sift", "SIFT1M (128-dim)", (90, 100)), ("glove", "GloVe-100 (100-dim)", (70, 93)),
          ("gist", "GIST1M (960-dim)", (80, 98)), ("sift1b", "SIFT1B (128-dim)", (85, 98.5))]
STYLE = {"DiskANN": dict(color="#A67C00", marker="o", linestyle="--"),
         "MCGI": dict(color="#5B3B9E", marker="s", linestyle="-")}

plt.rcParams.update({"font.size": 16, "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "lines.linewidth": 2, "lines.markersize": 6, "axes.labelsize": 17,
                     "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 15})

rows = list(csv.DictReader(open(os.path.join(HERE, "data", "recall_qps_clean.csv"))))
fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.3))
for ax, (ds, title, xlim) in zip(axes, PANELS):
    for method, st in STYLE.items():
        pts = sorted((float(r["recall"]), float(r["qps"])) for r in rows if r["dataset"] == ds and r["method"] == method)
        pts = [p for p in pts if p[0] >= xlim[0] - 2]
        ax.set_ylim(top=max(ax.get_ylim()[1], 1.05 * max(q for r, q in pts if r >= xlim[0])))
        ax.plot([p[0] for p in pts], [p[1] for p in pts], label=method, **st)
    ax.set_xlim(*xlim)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Recall@10 (%)")
    ax.set_title(title, fontsize=17)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4, integer=True))
    ax.grid(True, linestyle="--", alpha=0.3)
axes[0].set_ylabel("QPS (queries/s)")
axes[0].legend(loc="lower left", frameon=True, edgecolor="black")
fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
print(f"wrote {OUT}")
