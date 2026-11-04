"""Learned per-query L from features observable early in the search (offline simulation).

Features come from the results of a cheap first stage (L0 = smallest L, optionally the
second L as well): LID of the top-10 distances, d1, d10, and top-10 overlap between the
two stages. A ridge regression on even queries predicts log L*, where L* is the
smallest L in the grid at which the query reaches its best recall; the policy
L_q = L * exp(gamma * (pred - mean pred)) is evaluated on odd queries.
Usage: python3 adaptL_learn.py run_dir gt_file [recall_targets, e.g. 85,88,90]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adaptL_sim import parse, read_bin  # noqa: E402


TARGETS = [float(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [95, 98, 99, 99.5]


def lid_of(d):
    d = np.maximum(d.astype(np.float64), 1e-12)
    return 20.0 / np.maximum(np.log(d[:, -1:] / d).sum(1), 1e-9)


def main():
    run, gt_path = sys.argv[1], sys.argv[2]
    rows = parse(f"{run}/search.log")
    Ls = np.array([int(r[0]) for r in rows], dtype=float)
    io = np.array([r[2] for r in rows])
    with open(gt_path, "rb") as f:
        n, k = np.frombuffer(f.read(8), dtype=np.uint32)
        gt = np.frombuffer(f.read(int(n) * int(k) * 4), dtype=np.uint32).reshape(n, k)[:, :10]
    ids = [read_bin(f"{run}/res_{int(L)}_idx_uint32.bin", np.uint32)[:, :10] for L in Ls]
    dis = [read_bin(f"{run}/res_{int(L)}_dists_float.bin", np.float32)[:, :10] for L in Ls]
    rec = np.array([[len(np.intersect1d(a, b)) / 10 for a, b in zip(r, gt)] for r in ids]).T
    lstar = np.array([Ls[np.argmax(rec[i] >= rec[i].max())] for i in range(len(rec))])

    d0, d1 = dis[0].astype(np.float64), dis[1].astype(np.float64)
    overlap = np.array([len(np.intersect1d(a, b)) for a, b in zip(ids[0], ids[1])])
    feat_sets = {
        "lid": [np.log(lid_of(d0))],
        "lid+dist": [np.log(lid_of(d0)), np.log(d0[:, 0] + 1e-9), np.log(d0[:, -1] + 1e-9)],
        "lid+dist+overlap": [np.log(lid_of(d0)), np.log(d0[:, 0] + 1e-9), np.log(d0[:, -1] + 1e-9),
                             overlap, np.log(lid_of(d1)), np.log(d1[:, -1] / d0[:, -1] + 1e-9)],
    }
    q = np.arange(len(rec))
    fit, ev = q % 2 == 0, q % 2 == 1
    fixed = np.array([(rec[ev, i].mean() * 100, io[i]) for i in range(len(Ls))])
    for name, cols in feat_sets.items():
        X = np.column_stack(cols + [np.ones(len(rec))])
        y = np.log(lstar)
        w = np.linalg.solve(X[fit].T @ X[fit] + 1e-3 * np.eye(X.shape[1]), X[fit].T @ y[fit])
        pred = X @ w
        r2 = 1 - ((y[ev] - pred[ev]) ** 2).sum() / ((y[ev] - y[ev].mean()) ** 2).sum()
        best = None
        for g in (0.3, 0.5, 0.7, 1.0, 1.3):
            # stage-1 cost: queries pay at least the first-stage L when the overlap feature is used
            floor = Ls[1] if "overlap" in name else Ls[0]
            curve = []
            for L in np.geomspace(Ls[0], Ls[-1], 80):
                Lq = np.clip(L * np.exp(g * (pred[ev] - pred[fit].mean())), floor, Ls[-1])
                r = np.array([np.interp(Lq[j], Ls, rec[i]) for j, i in enumerate(q[ev])])
                curve.append((r.mean() * 100, np.interp(Lq, Ls, io).mean()))
            curve = np.array(sorted(curve))
            gains = []
            for t in TARGETS:
                ok = fixed[0, 0] <= t <= fixed[-1, 0] and curve[0, 0] <= t <= curve[-1, 0]
                gains.append(np.interp(t, fixed[:, 0], fixed[:, 1]) / np.interp(t, curve[:, 0], curve[:, 1])
                             if ok else np.nan)
            if best is None or np.nanmean(gains) > np.nanmean(best[1]):
                best = (g, gains)
        print(f"{name:18s} R2={r2:.3f} gamma={best[0]}  " + "  ".join(f"{x:.3f}" for x in best[1]))
        if name == "lid+dist":
            print(f"MCGI_ADAPT={w[0]:.5f},{w[1]:.5f},{w[2]:.5f},{w[3]:.5f},{best[0]},"
                  f"{pred[fit].mean():.5f},{int(round(io[0]))},8")


if __name__ == "__main__":
    main()
