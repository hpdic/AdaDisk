"""Offline simulation of a query-adaptive search list size L chosen from the query's LID.

Inputs: one search_disk_index run over an L list with result files kept (res_<L>_*.bin)
and its log (mean I/Os per L). Each query's LID is estimated from the distances of
its top-10 results at the smallest L. Queries are binned by LID quantile; per bin an L
is chosen on the even-indexed queries (greedy recall gain per I/O), and the policy is
evaluated on the odd-indexed queries against one fixed L at the same mean I/O.
Usage: python3 adaptL_sim.py run_dir gt_file
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m1_tune import parse  # noqa: E402

BINS = 8


def read_bin(path, dtype):
    with open(path, "rb") as f:
        n, k = np.frombuffer(f.read(8), dtype=np.uint32)
        return np.frombuffer(f.read(), dtype=dtype, count=int(n) * int(k)).reshape(n, k)


def main():
    run, gt_path = sys.argv[1], sys.argv[2]
    rows = parse(f"{run}/search.log")
    Ls = [int(r[0]) for r in rows]
    io = np.array([r[2] for r in rows])
    with open(gt_path, "rb") as f:
        n, k = np.frombuffer(f.read(8), dtype=np.uint32)
        gt = np.frombuffer(f.read(int(n) * int(k) * 4), dtype=np.uint32).reshape(n, k)[:, :10]
    rec = []
    for L in Ls:
        res = read_bin(f"{run}/res_{L}_idx_uint32.bin", np.uint32)[:, :10]
        rec.append([len(np.intersect1d(a, b)) / 10 for a, b in zip(res, gt)])
    rec = np.array(rec).T  # queries x L
    d = read_bin(f"{run}/res_{Ls[0]}_dists_float.bin", np.float32)[:, :10].astype(np.float64)
    d = np.maximum(d, 1e-12)
    lid = 20.0 / np.maximum(np.log(d[:, -1:] / d).sum(1), 1e-9)
    feats = {"lid": lid, "d10/d1": -(d[:, -1] / d[:, 0]),
             "random": np.random.default_rng(0).permutation(len(lid)).astype(float)}
    for name, f in feats.items():
        print(f"== feature {name}")
        simulate(f, Ls, io, rec)


def simulate(lid, Ls, io, rec):
    q = np.arange(len(lid))
    fit, ev = q % 2 == 0, q % 2 == 1
    edges = np.quantile(lid[fit], np.linspace(0, 1, BINS + 1)[1:-1])
    b = np.digitize(lid, edges)

    def policy_curve(mask):
        # greedy: start every bin at L index 0, repeatedly raise the bin with best gain per I/O
        idx = np.zeros(BINS, dtype=int)
        size = np.array([(mask & (b == j)).sum() for j in range(BINS)])
        mrec = np.array([[rec[mask & (b == j), i].mean() for i in range(len(Ls))] for j in range(BINS)])
        out = [idx.copy()]
        while True:
            best, bj = 0, -1
            for j in range(BINS):
                for i in range(idx[j] + 1, len(Ls)):  # look ahead past flat steps
                    g = (mrec[j, i] - mrec[j, idx[j]]) * size[j] / ((io[i] - io[idx[j]]) * size[j])
                    if g > best:
                        best, bj, bi = g, j, i
            if bj < 0:
                break
            idx[bj] = bi
            out.append(idx.copy())
        return out

    steps = policy_curve(fit)
    pol = []
    for idx in steps:
        m_io = np.mean([io[idx[b[i]]] for i in q[ev]])
        m_rec = np.mean([rec[i, idx[b[i]]] for i in q[ev]])
        pol.append((m_rec * 100, m_io))
    pol = np.array(sorted(pol))
    fixed = np.array([(rec[ev, i].mean() * 100, io[i]) for i in range(len(Ls))])
    print("target  fixed_io  adaptive_io  gain")
    for t in (90, 95, 97, 98, 99, 99.5):
        if fixed[0, 0] <= t <= fixed[-1, 0] and pol[0, 0] <= t <= pol[-1, 0]:
            fi = np.interp(t, fixed[:, 0], fixed[:, 1])
            ai = np.interp(t, pol[:, 0], pol[:, 1])
            print(f"{t:6.1f}  {fi:8.1f}  {ai:11.1f}  {fi / ai:.3f}")


if __name__ == "__main__":
    main()


def parametric(run, gt_path, gammas=(0.5, 1.0, 1.5, 2.0, 3.0)):
    """L_q = L * (lid_q / median lid)^gamma, per-query recall and I/O interpolated over the L grid."""
    rows = parse(f"{run}/search.log")
    Ls = np.array([int(r[0]) for r in rows], dtype=float)
    io = np.array([r[2] for r in rows])
    with open(gt_path, "rb") as f:
        n, k = np.frombuffer(f.read(8), dtype=np.uint32)
        gt = np.frombuffer(f.read(int(n) * int(k) * 4), dtype=np.uint32).reshape(n, k)[:, :10]
    rec = np.array([[len(np.intersect1d(a, b)) / 10 for a, b in
                     zip(read_bin(f"{run}/res_{int(L)}_idx_uint32.bin", np.uint32)[:, :10], gt)] for L in Ls]).T
    d = np.maximum(read_bin(f"{run}/res_{int(Ls[0])}_dists_float.bin", np.float32)[:, :10].astype(np.float64), 1e-12)
    lid = 20.0 / np.maximum(np.log(d[:, -1:] / d).sum(1), 1e-9)
    rel = lid / np.median(lid)
    fixed = np.array([(rec[:, i].mean() * 100, io[i]) for i in range(len(Ls))])
    print("gamma  " + "  ".join(f"g@{t}" for t in (95, 98, 99, 99.5)))
    for g in gammas:
        curve = []
        for L in np.geomspace(Ls[0], Ls[-1], 60):
            Lq = np.clip(L * rel ** g, Ls[0], Ls[-1])
            r = np.array([np.interp(Lq[i], Ls, rec[i]) for i in range(len(Lq))])
            curve.append((r.mean() * 100, np.interp(Lq, Ls, io).mean()))
        curve = np.array(sorted(curve))
        cells = []
        for t in (95, 98, 99, 99.5):
            ok = fixed[0, 0] <= t <= fixed[-1, 0] and curve[0, 0] <= t <= curve[-1, 0]
            cells.append(f"{np.interp(t, fixed[:, 0], fixed[:, 1]) / np.interp(t, curve[:, 0], curve[:, 1]):.3f}"
                         if ok else "  nan")
        print(f"{g:5.1f}  " + "  ".join(cells))
