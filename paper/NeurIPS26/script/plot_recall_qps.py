"""Recall-QPS curves of DiskANN and MCGI: SIFT1M, GloVe-100, GIST1M (figures/recall_qps.pdf) and SIFT1B, Deep1B
(figures/recall_qps_1b.pdf).

Data: data/recall_qps_clean.csv, exported from the clean serial runs (one search per method, page cache
dropped, idle machine). Both methods search the same index; MCGI adapts the search list per query.
"""
import csv
import os

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
FIGURES = {"recall_qps.pdf": [("sift", "SIFT1M (128-dim)", (90, 100)), ("glove", "GloVe-100 (100-dim)", (70, 93)),
                              ("gist", "GIST1M (960-dim)", (80, 98))],
           "recall_qps_1b.pdf": [("sift1b", "SIFT1B (128-dim)", (85, 98.5)), ("deep1b", "Deep1B (96-dim)", (85, 98.5))]}
STYLE = {"DiskANN": dict(color="#A67C00", marker="o", linestyle="--"),
         "MCGI": dict(color="#5B3B9E", marker="s", linestyle="-")}

plt.rcParams.update({"font.size": 16, "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "lines.linewidth": 2, "lines.markersize": 6, "axes.labelsize": 17,
                     "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 15})

rows = list(csv.DictReader(open(os.path.join(HERE, "data", "recall_qps_clean.csv"))))
for out, panels in FIGURES.items():
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels) + 0.4, 3.3 if len(panels) == 2 else 2.9))
    for ax, (ds, title, xlim) in zip(axes, panels):
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
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True, edgecolor="black", facecolor="white",
               framealpha=1.0, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(os.path.join(FIG, out), bbox_inches="tight")
    print(f"wrote {out}")
