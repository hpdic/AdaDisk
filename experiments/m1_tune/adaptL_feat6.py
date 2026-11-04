"""Compare feature sets for the query-adaptive L on a 6-feature MCGI_FEAT_DUMP run (checkpoint 0).

Features: 0 log LID (10 expanded), 1 log d1, 2 log d10, 3 log PQ-noise/contrast,
4 log LID of the 64 best candidates, 5 log(1 + candidates within 1.2 d10).
Ridge fit on even queries, policy evaluated on odd queries with per-query I/Os.
Usage: python3 adaptL_feat6.py run_dir gt_file recall_targets
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adaptL_sim import parse, read_bin  # noqa: E402

SETS = {"lid+dist (current)": [0, 1, 2], "+cand LID": [0, 1, 2, 4], "+crowd": [0, 1, 2, 5],
        "+cand LID+crowd": [0, 1, 2, 4, 5], "all 6": [0, 1, 2, 3, 4, 5]}


def main():
    run, gt_path = sys.argv[1], sys.argv[2]
    targets = [float(x) for x in sys.argv[3].split(",")]
    Ls = np.array([int(r[0]) for r in parse(f"{run}/search.log")], dtype=float)
    with open(gt_path, "rb") as f:
        n, k = np.frombuffer(f.read(8), dtype=np.uint32)
        gt = np.frombuffer(f.read(int(n) * int(k) * 4), dtype=np.uint32).reshape(n, k)[:, :10]
    rec = np.array([[len(np.intersect1d(a, b)) / 10 for a, b in
                     zip(read_bin(f"{run}/res_{int(L)}_idx_uint32.bin", np.uint32)[:, :10], gt)] for L in Ls]).T
    st = [read_bin(f"{run}/res_{int(L)}_stats.bin", np.float32) for L in Ls]
    io = np.array([s[:, 0] for s in st]).T
    F = st[-1][:, 1:7].astype(np.float64)
    y = np.log([Ls[np.argmax(rec[i] >= rec[i].max())] for i in range(len(rec))])
    q = np.arange(len(y))
    fit, ev = q % 2 == 0, q % 2 == 1
    fixed = np.array([(rec[ev, i].mean() * 100, io[ev, i].mean()) for i in range(len(Ls))])
    for name, cols in SETS.items():
        X = np.column_stack([F[:, cols], np.ones(len(y))])
        w = np.linalg.solve(X[fit].T @ X[fit] + 1e-3 * np.eye(X.shape[1]), X[fit].T @ y[fit])
        pred = X @ w
        r2 = 1 - ((y[ev] - pred[ev]) ** 2).sum() / ((y[ev] - y[ev].mean()) ** 2).sum()
        best = None
        for g in (0.5, 0.7, 1.0, 1.3):
            c = []
            for L in np.geomspace(Ls[0], Ls[-1], 60):
                Lq = np.clip(L * np.exp(g * (pred[ev] - pred[fit].mean())), Ls[0], Ls[-1])
                c.append((np.mean([np.interp(Lq[j], Ls, rec[i]) for j, i in enumerate(q[ev])]) * 100,
                          np.mean([np.interp(Lq[j], Ls, io[i]) for j, i in enumerate(q[ev])])))
            c = np.array(sorted(c))
            gains = [np.interp(t, fixed[:, 0], fixed[:, 1]) / np.interp(t, c[:, 0], c[:, 1])
                     if fixed[0, 0] <= t <= fixed[-1, 0] and c[0, 0] <= t <= c[-1, 0] else np.nan for t in targets]
            if best is None or np.nanmean(gains) > np.nanmean(best[1]):
                best = (g, gains)
        print(f"{name:20s} R2={r2:.3f} gamma={best[0]}  " + "  ".join(f"{t:g}:{x:.3f}" for t, x in zip(targets, best[1])),
              flush=True)


if __name__ == "__main__":
    main()
