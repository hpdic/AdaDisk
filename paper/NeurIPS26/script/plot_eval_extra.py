"""Comparison, ablation, and latency figures of the evaluation.

figures/gist_compare.pdf  Recall-QPS of DiskANN, PipeANN, and MCGI on the same GIST1M index (R = 32 and 48)
figures/ablation.pdf      QPS gain of MCGI variants over DiskANN on the same index, at the highest target recall
figures/latency.pdf       recall against the requested list size, and mean latency against recall (appendix)
Data: data/gist_compare.csv, data/ablation.csv, data/latency.csv (experiments/baselines/export_eval_extra.py).
"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
STYLE = {"DiskANN": dict(color="#A67C00", marker="o", linestyle="--"),
         "PipeANN": dict(color="#2E7D6B", marker="^", linestyle="-."),
         "MCGI": dict(color="#5B3B9E", marker="s", linestyle="-")}
DS = {"sift": ("SIFT1M", 99), "glove": ("GloVe-100", 90), "gist": ("GIST1M", 95)}

plt.rcParams.update({"font.size": 16, "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "lines.linewidth": 2, "lines.markersize": 6, "axes.labelsize": 17,
                     "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 15})


def load(name):
    return list(csv.DictReader(open(os.path.join(HERE, "data", name))))


def curve(rows, x="recall", y="qps", **match):
    pts = sorted((float(r[x]), float(r[y])) for r in rows if all(r[k] == str(v) for k, v in match.items()))
    return np.array(pts)


def at(c, t):
    return float(np.interp(t, c[:, 0], c[:, 1])) if c[0, 0] <= t <= c[-1, 0] else float("nan")


def top_legend(fig, axes, ncol, top):
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=ncol, frameon=True, edgecolor="black", facecolor="white",
               framealpha=1.0, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, top))


def gist_compare():
    rows = load("gist_compare.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.3))
    for ax, R in zip(axes, (32, 48)):
        for method, key in (("DiskANN", "DiskANN"), ("PipeANN", "PipeANN-W4"), ("MCGI", "MCGI")):
            c = curve(rows, R=R, method=key)
            c = c[c[:, 0] >= 83]
            ax.plot(c[:, 0], c[:, 1], label=method, **STYLE[method])
        ax.set_xlim(85, 99.5 if R == 48 else 98)
        ax.set_ylim(bottom=0)
        ax.set_title(f"GIST1M, $R = {R}$", fontsize=17)
        ax.set_xlabel("Recall@10 (%)")
        ax.xaxis.set_major_locator(plt.MaxNLocator(4, integer=True))
        ax.grid(True, linestyle="--", alpha=0.3)
    axes[0].set_ylabel("QPS (queries/s)")
    top_legend(fig, axes, 3, 0.86)
    fig.savefig(os.path.join(FIG, "gist_compare.pdf"), bbox_inches="tight")


def ablation():
    rows = load("ablation.csv")
    gain = {}
    for ds, (_, t) in DS.items():
        base = at(curve(rows, dataset=ds, variant="stock"), t)
        for v in {r["variant"] for r in rows if r["dataset"] == ds}:
            gain[ds, v] = at(curve(rows, dataset=ds, variant=v), t) / base
    gamma_main = {"sift": 0.7, "glove": 1.0, "gist": 0.7}
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 2.9), gridspec_kw={"width_ratios": [1.4, 1, 1]})
    variants = [("full", "MCGI", "#5B3B9E"), ("lid", "LID only", "#9C86C9"), ("nolid", "No LID", "#8C8C8C"),
                ("random", "Random", "#D9B45A")]
    x = np.arange(len(DS))
    for i, (v, label, color) in enumerate(variants):
        axes[0].bar(x + (i - 1.5) * 0.2, [gain[ds, v] for ds in DS], 0.2, label=label, color=color,
                    edgecolor="black", linewidth=0.6)
    axes[0].set_xticks(x, [DS[ds][0] for ds in DS])
    axes[0].set_ylabel("QPS gain")
    axes[0].set_ylim(0.6, 1.55)

    marks = {"sift": "o", "glove": "s", "gist": "^"}
    for ds in DS:
        g = [(gm, gain[ds, "full"] if gm == gamma_main[ds] else gain[ds, f"g{gm:g}"]) for gm in (0.5, 0.7, 1.0, 1.3)]
        axes[1].plot([a for a, _ in g], [b for _, b in g], marker=marks[ds], label=DS[ds][0], color="#5B3B9E",
                     linestyle="-", alpha={"sift": 1.0, "glove": 0.65, "gist": 0.4}[ds])
        n = [(12, gain[ds, "n12"]), (24, gain[ds, "full"]), (48, gain[ds, "n48"])]
        axes[2].plot([a for a, _ in n], [b for _, b in n], marker=marks[ds], label=DS[ds][0], color="#5B3B9E",
                     linestyle="-", alpha={"sift": 1.0, "glove": 0.65, "gist": 0.4}[ds])
    axes[1].set_xlabel(r"Strength $\gamma$")
    axes[1].set_xticks([0.5, 0.7, 1.0, 1.3])
    axes[2].set_xlabel(r"Checkpoint $n_0$")
    axes[2].set_xticks([12, 24, 48])
    for ax in axes[1:]:
        ax.set_ylim(1.0, 1.55)

    for ax in axes:
        ax.axhline(1.0, color="black", linewidth=1)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    kw = dict(frameon=True, edgecolor="black", facecolor="white", framealpha=1.0, fontsize=13, columnspacing=0.7,
              handlelength=1.2)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=4, **kw)
    fig.legend(*axes[1].get_legend_handles_labels(), loc="upper right", bbox_to_anchor=(1.0, 1.0), ncol=3, **kw)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(os.path.join(FIG, "ablation.pdf"), bbox_inches="tight")
    for k in sorted(gain):
        print(k, round(gain[k], 3))


def latency():
    rows = load("latency.csv")
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.3))
    for method in ("DiskANN", "MCGI"):
        c = curve(rows, x="L", y="recall", dataset="gist", method=method)
        axes[0].plot(c[:, 0], c[:, 1], label=method, **STYLE[method])
    axes[0].set_xscale("log")
    axes[0].set_xlabel(r"List size $\bar{L}$")
    axes[0].set_ylabel("Recall@10 (%)")
    axes[0].set_title("GIST1M", fontsize=17)
    for ax, ds, title, lo in zip(axes[1:], ("gist", "sift1b", "deep1b"), ("GIST1M", "SIFT1B", "Deep1B"), (85, 85, 85)):
        for method in ("DiskANN", "MCGI"):
            c = curve(rows, y="mean_ms", dataset=ds, method=method)
            c = c[c[:, 0] >= lo - 1]
            ax.plot(c[:, 0], c[:, 1], label=method, **STYLE[method])
        ax.set_xlim(lo, None)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Recall@10 (%)")
        ax.set_title(title, fontsize=17)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4, integer=True))
    axes[1].set_ylabel("Mean latency (ms)")
    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.3)
    top_legend(fig, axes, 2, 0.86)
    fig.savefig(os.path.join(FIG, "latency.pdf"), bbox_inches="tight")


if __name__ == "__main__":
    gist_compare()
    ablation()
    latency()
