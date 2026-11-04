"""Fit the query-adaptive L on in-search checkpoint features (MCGI_FEAT_DUMP runs).

Reads res_<L>_idx_uint32.bin and res_<L>_stats.bin (per-query I/Os and the checkpoint
features log LID, log d1, log d10, log PQ-noise/contrast; taken from the largest L).
A ridge regression on even queries predicts log L* (smallest L reaching the query's
best recall); the policy L_q = L * exp(gamma * (pred - mean)) is evaluated on odd
queries with per-query I/Os interpolated over the L grid.
Prints the MCGI_ADAPT string (9 values) for the best feature set.
Usage: python3 adaptL_feat.py run_dir gt_file recall_targets n0
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adaptL_sim import parse, read_bin  # noqa: E402


def main():
    run, gt_path = sys.argv[1], sys.argv[2]
    targets = [float(x) for x in sys.argv[3].split(",")]
    n0 = int(sys.argv[4])
    Ls = np.array([int(r[0]) for r in parse(f"{run}/search.log")], dtype=float)
    with open(gt_path, "rb") as f:
        n, k = np.frombuffer(f.read(8), dtype=np.uint32)
        gt = np.frombuffer(f.read(int(n) * int(k) * 4), dtype=np.uint32).reshape(n, k)[:, :10]
    rec = np.array([[len(np.intersect1d(a, b)) / 10 for a, b in
                     zip(read_bin(f"{run}/res_{int(L)}_idx_uint32.bin", np.uint32)[:, :10], gt)] for L in Ls]).T
    st = [read_bin(f"{run}/res_{int(L)}_stats.bin", np.float32) for L in Ls]
    io = np.array([s[:, 0] for s in st]).T  # queries x L
    F = st[-1][:, 1:].astype(np.float64)
    lstar = np.array([Ls[np.argmax(rec[i] >= rec[i].max())] for i in range(len(rec))])
    y = np.log(lstar)
    q = np.arange(len(y))
    fit, ev = q % 2 == 0, q % 2 == 1
    fixed = np.array([(rec[ev, i].mean() * 100, io[ev, i].mean()) for i in range(len(Ls))])
    sets = {"lid": [0], "lid+dist": [0, 1, 2], "lid+dist+noise": [0, 1, 2, 3], "noise": [3]}
    best_all = None
    for name, cols in sets.items():
        X = np.column_stack([F[:, cols], np.ones(len(y))])
        w = np.linalg.solve(X[fit].T @ X[fit] + 1e-3 * np.eye(X.shape[1]), X[fit].T @ y[fit])
        pred = X @ w
        r2 = 1 - ((y[ev] - pred[ev]) ** 2).sum() / ((y[ev] - y[ev].mean()) ** 2).sum()
        best = None
        for g in (0.3, 0.5, 0.7, 1.0, 1.3, 1.6):
            curve = []
            for L in np.geomspace(Ls[0], Ls[-1], 80):
                Lq = np.clip(L * np.exp(g * (pred[ev] - pred[fit].mean())), Ls[0], Ls[-1])
                r = np.array([np.interp(Lq[j], Ls, rec[i]) for j, i in enumerate(q[ev])])
                c = np.array([np.interp(Lq[j], Ls, io[i]) for j, i in enumerate(q[ev])])
                curve.append((r.mean() * 100, c.mean()))
            curve = np.array(sorted(curve))
            gains = []
            for t in targets:
                ok = fixed[0, 0] <= t <= fixed[-1, 0] and curve[0, 0] <= t <= curve[-1, 0]
                gains.append(np.interp(t, fixed[:, 0], fixed[:, 1]) / np.interp(t, curve[:, 0], curve[:, 1])
                             if ok else np.nan)
            if best is None or np.nanmean(gains) > np.nanmean(best[1]):
                best = (g, gains)
        print(f"{name:16s} R2={r2:.3f} gamma={best[0]}  " + "  ".join(f"{x:.3f}" for x in best[1]), flush=True)
        full = np.zeros(4)
        full[cols] = w[:-1]
        if best_all is None or np.nanmean(best[1]) > best_all[0]:
            best_all = (np.nanmean(best[1]), full, w[-1], best[0], pred[fit].mean())
    _, full, b, g, pm = best_all
    print("MCGI_ADAPT=" + ",".join(f"{x:.5f}" for x in full) + f",{b:.5f},{g},{pm:.5f},{n0},8")


if __name__ == "__main__":
    main()
