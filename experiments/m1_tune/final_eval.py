"""Final protocol for the query-adaptive L (MCGI search) on SIFT1M / GloVe-100 / GIST1M.

1. Calibration: one MCGI_FEAT_DUMP search of the calibration queries (calib_prep.py, disjoint
   from the official queries) over the L grid; a ridge model on the checkpoint features
   (log LID, log d1, log d10, log candidate LID, log crowding) predicts log L*, gamma is chosen
   on the calibration set.
2. Test: real searches of all official queries on the same index, stock DiskANN vs MCGI_ADAPT,
   run concurrently so they see the same disk; QPS and mean I/Os at matched recall.
Usage: python3 final_eval.py ds index_name [index_name ...]   (indices under ~/data/mcgi/rl/<ds>)
"""
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adaptL_sim import parse, read_bin  # noqa: E402
from m1_tune import DS, ENV, HOME, at_recall, paths  # noqa: E402

RL = f"{HOME}/data/mcgi/rl"
OUT = f"{HOME}/data/mcgi/final2"
FEATS = [0, 1, 2, 4, 5]  # log LID, log d1, log d10, log candidate LID, log(1 + crowd); see QueryStats
SEARCH = f"{HOME}/hpdic/adadisk/build/apps/search_disk_index"
L_LIST = "10 15 20 25 30 40 50 60 80 100 150 200 300 400".split()
N0 = 24
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def run_search(idx, query, gt, out, env, tag):
    cmd = [SEARCH, "--data_type", "float", "--dist_fn", "l2", "--index_path_prefix", idx,
           "--query_file", query, "--gt_file", gt, "-K", "10", "-L", *L_LIST,
           "--result_path", f"{out}/{tag}", "-T", "32"]
    log = open(f"{out}/{tag}.log", "w")
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=dict(ENV, **env))


def fit(out, calib_gt):
    Ls = np.array([int(r[0]) for r in parse(f"{out}/calib.log")], dtype=float)
    with open(calib_gt, "rb") as f:
        n, k = np.frombuffer(f.read(8), dtype=np.uint32)
        gt = np.frombuffer(f.read(int(n) * int(k) * 4), dtype=np.uint32).reshape(n, k)[:, :10]
    rec = np.array([[len(np.intersect1d(a, b)) / 10 for a, b in
                     zip(read_bin(f"{out}/calib_{int(L)}_idx_uint32.bin", np.uint32)[:, :10], gt)] for L in Ls]).T
    st = [read_bin(f"{out}/calib_{int(L)}_stats.bin", np.float32) for L in Ls]
    io = np.array([s[:, 0] for s in st]).T
    F = st[-1][:, [1 + c for c in FEATS]].astype(np.float64)
    y = np.log([Ls[np.argmax(rec[i] >= rec[i].max())] for i in range(len(rec))])
    X = np.column_stack([F, np.ones(len(y))])
    w = np.linalg.solve(X.T @ X + 1e-3 * np.eye(X.shape[1]), X.T @ y)
    pred = X @ w
    fixed = np.array([(rec[:, i].mean() * 100, io[:, i].mean()) for i in range(len(Ls))])
    lo, hi = max(fixed[0, 0], 80.0), fixed[-1, 0]
    best = None
    for g in (0.5, 0.7, 1.0, 1.3):
        c = []
        for L in np.geomspace(Ls[0], Ls[-1], 60):
            Lq = np.clip(L * np.exp(g * (pred - pred.mean())), Ls[0], Ls[-1])
            c.append((np.mean([np.interp(Lq[i], Ls, rec[i]) for i in range(len(Lq))]) * 100,
                      np.mean([np.interp(Lq[i], Ls, io[i]) for i in range(len(Lq))])))
        c = np.array(sorted(c))
        ts = np.linspace(lo, min(hi, c[-1, 0]), 8)
        score = np.mean(np.interp(ts, fixed[:, 0], fixed[:, 1]) / np.interp(ts, c[:, 0], c[:, 1]))
        if best is None or score > best[0]:
            best = (score, g)
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    full = np.zeros(6)
    full[FEATS] = w[:-1]
    adapt = ",".join(f"{x:.5f}" for x in full) + f",{w[-1]:.5f},{best[1]},{pred.mean():.5f},{N0},8"
    return adapt, r2, best[0]


def main():
    ds, names = sys.argv[1], sys.argv[2:]
    p = paths(ds)
    cq, cg = f"{DATA}/{ds}/{ds}_calib_query.bin", f"{DATA}/{ds}/{ds}_calib_gt.bin"
    lines = []
    for n in names:
        out = f"{OUT}/{ds}/{n}"
        os.makedirs(out, exist_ok=True)
        idx = f"{RL}/{ds}/{n}/idx"
        if not (os.path.exists(f"{out}/calib.log") and len(parse(f"{out}/calib.log")) == len(L_LIST)):
            run_search(idx, cq, cg, out, {"MCGI_FEAT_DUMP": "1", "MCGI_FEAT_N0": str(N0)}, "calib").wait()
        adapt, r2, cal_gain = fit(out, cg)
        with open(f"{out}/adapt.txt", "w") as f:
            f.write(f"MCGI_ADAPT={adapt}\ncalibration R2={r2:.3f} gain={cal_gain:.3f}\n")
        procs = [run_search(idx, p["query"], p["gt"], out, {}, "stock"),
                 run_search(idx, p["query"], p["gt"], out, {"MCGI_ADAPT": adapt}, "mcgi")]
        for pr in procs:
            pr.wait()
        subprocess.run(f"rm -f {out}/stock_* {out}/mcgi_* {out}/calib_*_dists_float.bin", shell=True)
        s, m = np.array(parse(f"{out}/stock.log")), np.array(parse(f"{out}/mcgi.log"))
        cells = [n, f"R2={r2:.3f}"]
        for t in DS[ds]["targets"]:
            cells.append(f"{t:g}: io {at_recall(s, t, 2) / at_recall(m, t, 2):.3f} "
                         f"qps {at_recall(m, t, 1) / at_recall(s, t, 1):.3f}")
        lines.append("  ".join(cells))
        print(lines[-1], flush=True)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"final2_{ds}.txt"), "a") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
