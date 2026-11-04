"""Sequential query-adaptive L: re-estimate the list size at checkpoints n0, 2 n0, 4 n0 expansions.

Per checkpoint k, a ridge model on even queries predicts log L* from the features of
checkpoints 0..k (log LID, log d1, log d10, log PQ-noise/contrast, so later checkpoints
also see how LID and the distances moved). Policy on odd queries: at checkpoint 0 set
L_0; if the search under L_0 runs past the next checkpoint, re-predict and replace the
list size (never below the expansions already spent). Per-query recall and I/Os are
interpolated over the L grid of one MCGI_FEAT_DUMP run. Expansions are approximated by
I/Os (no node cache).
Usage: python3 adaptL_seq.py run_dir gt_file recall_targets n0
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adaptL_sim import parse, read_bin  # noqa: E402


def ridge(X, y):
    X1 = np.column_stack([X, np.ones(len(X))])
    w = np.linalg.solve(X1.T @ X1 + 1e-3 * np.eye(X1.shape[1]), X1.T @ y)
    return lambda Z: np.column_stack([Z, np.ones(len(Z))]) @ w


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
    io = np.array([s[:, 0] for s in st]).T
    F = st[-1][:, 1:13].astype(np.float64)  # 3 checkpoints x 4 features
    y = np.log([Ls[np.argmax(rec[i] >= rec[i].max())] for i in range(len(rec))])
    q = np.arange(len(y))
    fit, ev = q % 2 == 0, q % 2 == 1
    fixed = np.array([(rec[ev, i].mean() * 100, io[ev, i].mean()) for i in range(len(Ls))])
    models = [ridge(F[fit][:, :4 * (k + 1)], y[fit]) for k in range(3)]
    preds = [m(F[:, :4 * (k + 1)]) for k, m in enumerate(models)]
    for k in range(3):
        p = preds[k][ev]
        r2 = 1 - ((y[ev] - p) ** 2).sum() / ((y[ev] - y[ev].mean()) ** 2).sum()
        print(f"checkpoint {k} ({n0 << k} expansions): R2={r2:.3f}")

    def L_at_io(i, budget):  # smallest list size whose I/O reaches the budget
        return float(np.interp(budget, io[i], Ls))

    def run_policy(L, g, stages):
        r, c = [], []
        for i in q[ev]:
            Lq = L * np.exp(g * (preds[0][i] - preds[0][fit].mean()))
            for k in range(1, stages):
                if np.interp(Lq, Ls, io[i]) <= (n0 << k):
                    break
                floor = L_at_io(i, n0 << k)
                Lq = max(floor, L * np.exp(g * (preds[k][i] - preds[k][fit].mean())))
            Lq = np.clip(Lq, Ls[0], Ls[-1])
            r.append(np.interp(Lq, Ls, rec[i]))
            c.append(np.interp(Lq, Ls, io[i]))
        return np.mean(r) * 100, np.mean(c)

    for stages in (1, 2, 3):
        best = None
        for g in (0.5, 0.7, 1.0, 1.3):
            curve = np.array(sorted(run_policy(L, g, stages) for L in np.geomspace(Ls[0], Ls[-1], 60)))
            gains = []
            for t in targets:
                ok = fixed[0, 0] <= t <= fixed[-1, 0] and curve[0, 0] <= t <= curve[-1, 0]
                gains.append(np.interp(t, fixed[:, 0], fixed[:, 1]) / np.interp(t, curve[:, 0], curve[:, 1])
                             if ok else np.nan)
            if best is None or np.nanmean(gains) > np.nanmean(best[1]):
                best = (g, gains)
        print(f"{stages} checkpoint(s) gamma={best[0]}  " + "  ".join(f"{t:g}:{x:.3f}" for t, x in zip(targets, best[1])))


if __name__ == "__main__":
    main()
