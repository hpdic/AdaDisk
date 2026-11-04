"""R x L_build sweep on SIFT1M / GloVe-100: stock DiskANN vs offline MCGI, fixed vs query-adaptive L.

Every index is built one-shot (as in run_exp.sh) and searched once over the L grid with
MCGI_FEAT_DUMP, so the per-query I/Os, results and checkpoint features of that single run
give both the fixed-L curve and the query-adaptive policy (fit on even queries, evaluated
on odd queries; see adaptL_feat.py). MCGI indices reuse the PQ of the stock index with the
same R and L.
Usage: python3 rl_sweep.py sift|glove|gist
"""
import concurrent.futures as cf
import os
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adaptL_sim import parse, read_bin  # noqa: E402
from m1_tune import BUILD, DS, ENV, HOME, paths  # noqa: E402

OUT = f"{HOME}/data/mcgi/rl"
SEARCH = f"{HOME}/hpdic/adadisk/build/apps/search_disk_index"
LBS = [50, 100, 200]
# GIST: a 960-dim float node with R = 64 spans two 4 KB sectors; PQ at 300 bytes instead of 100
RS = {"sift": [32, 48, 64], "glove": [32, 48, 64], "gist": [32, 48]}
PQ_GB = {"sift": "0.1", "glove": "0.1", "gist": "0.3"}
METHODS = {"diskann": None, "mcgi": (1.0, 1.5)}
L_LIST = "10 15 20 25 30 40 50 60 80 100 150 200 300 400".split()
N0 = 24


def name(m, R, Lb):
    return f"{m}_R{R}_L{Lb}"


def build(ds, m, R, Lb, threads):
    d = f"{OUT}/{ds}/{name(m, R, Lb)}"
    os.makedirs(d, exist_ok=True)
    p = paths(ds)
    cmd = [BUILD, "--data_type", "float", "--dist_fn", "l2", "--data_path", p["base"],
           "--index_path_prefix", f"{d}/idx", "-R", str(R), "-L", str(Lb),
           "-B", PQ_GB[ds], "-M", "4.0", "-T", str(threads)]
    base = f"{OUT}/{ds}/{name('diskann', R, Lb)}/idx"
    if METHODS[m] is not None:
        a, b = METHODS[m]
        cmd += ["--use_mcgi", "--lid_path", p["lid"], "--alpha_min", str(a), "--alpha_max", str(b),
                "--codebook_prefix", base]
    with open(f"{d}/build.log", "w") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=ENV).returncode
    if rc == 0 and METHODS[m] is not None:
        for suf in ("_pq_pivots.bin", "_pq_compressed.bin"):
            shutil.copy(f"{base}{suf}", f"{d}/idx{suf}")
    return rc


def search(ds, n):
    d = f"{OUT}/{ds}/{n}"
    if os.path.exists(f"{d}/search.log") and len(parse(f"{d}/search.log")) == len(L_LIST):
        return
    p = paths(ds)
    cmd = [SEARCH, "--data_type", "float", "--dist_fn", "l2", "--index_path_prefix", f"{d}/idx",
           "--query_file", p["query"], "--gt_file", p["gt"], "-K", "10", "-L", *L_LIST,
           "--result_path", f"{d}/res", "-T", "32"]
    env = dict(ENV, MCGI_FEAT_DUMP="1", MCGI_FEAT_N0=str(N0))
    with open(f"{d}/search.log", "w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    subprocess.run(f"rm -f {d}/res_*_dists_float.bin", shell=True)


def analyze(ds, n, gt):
    """Mean I/Os at the recall targets on odd queries: fixed L and query-adaptive L."""
    d = f"{OUT}/{ds}/{n}"
    Ls = np.array([int(r[0]) for r in parse(f"{d}/search.log")], dtype=float)
    rec = np.array([[len(np.intersect1d(a, b)) / 10 for a, b in
                     zip(read_bin(f"{d}/res_{int(L)}_idx_uint32.bin", np.uint32)[:, :10], gt)] for L in Ls]).T
    st = [read_bin(f"{d}/res_{int(L)}_stats.bin", np.float32) for L in Ls]
    io = np.array([s[:, 0] for s in st]).T
    F = st[-1][:, 1:4].astype(np.float64)  # log LID, log d1, log d10
    y = np.log([Ls[np.argmax(rec[i] >= rec[i].max())] for i in range(len(rec))])
    q = np.arange(len(y))
    fit, ev = q % 2 == 0, q % 2 == 1
    X = np.column_stack([F, np.ones(len(y))])
    w = np.linalg.solve(X[fit].T @ X[fit] + 1e-3 * np.eye(4), X[fit].T @ y[fit])
    pred = X @ w
    fixed = np.array([(rec[ev, i].mean() * 100, io[ev, i].mean()) for i in range(len(Ls))])
    curves = {}
    for g in (0.5, 0.7, 1.0):
        c = []
        for L in np.geomspace(Ls[0], Ls[-1], 80):
            Lq = np.clip(L * np.exp(g * (pred[ev] - pred[fit].mean())), Ls[0], Ls[-1])
            c.append((np.mean([np.interp(Lq[j], Ls, rec[i]) for j, i in enumerate(q[ev])]) * 100,
                      np.mean([np.interp(Lq[j], Ls, io[i]) for j, i in enumerate(q[ev])])))
        curves[g] = np.array(sorted(c))

    def at(curve, t):
        return np.interp(t, curve[:, 0], curve[:, 1]) if curve[0, 0] <= t <= curve[-1, 0] else np.nan
    out = {}
    for t in DS[ds]["targets"]:
        out[t] = (at(fixed, t), np.nanmin([at(c, t) for c in curves.values()]))
    return out


def main():
    ds = sys.argv[1]
    jobs = [(m, R, Lb) for m in METHODS for R in RS[ds] for Lb in LBS]
    # stock builds first: MCGI builds take their PQ codebooks
    for group in ([j for j in jobs if j[0] == "diskann"], [j for j in jobs if j[0] != "diskann"]):
        todo = [j for j in group if not os.path.exists(f"{OUT}/{ds}/{name(*j)}/idx_disk.index")]
        with cf.ThreadPoolExecutor(max(1, len(todo))) as ex:
            rcs = list(ex.map(lambda j: build(ds, *j, max(16, 192 // max(1, len(todo)))), todo))
        for j, rc in zip(todo, rcs):
            if rc != 0:
                print(f"build failed: {j}", flush=True)
    print("builds done", flush=True)
    names = [name(*j) for j in jobs if os.path.exists(f"{OUT}/{ds}/{name(*j)}/idx_disk.index")]
    with cf.ThreadPoolExecutor(6) as ex:
        list(ex.map(lambda n: search(ds, n), names))

    with open(paths(ds)["gt"], "rb") as f:
        nq, k = np.frombuffer(f.read(8), dtype=np.uint32)
        gt = np.frombuffer(f.read(int(nq) * int(k) * 4), dtype=np.uint32).reshape(nq, k)[:, :10]
    targets = DS[ds]["targets"]
    ref = analyze(ds, name("diskann", 32, DS[ds]["L"]), gt)
    lines = ["config," + ",".join(f"io_fixed@{t:g},io_adapt@{t:g},gain_fixed@{t:g},gain_adapt@{t:g}"
                                  for t in targets)]
    for n in names:
        r = analyze(ds, n, gt)
        cells = [n]
        for t in targets:
            cells += [f"{r[t][0]:.1f}", f"{r[t][1]:.1f}", f"{ref[t][0] / r[t][0]:.3f}", f"{ref[t][0] / r[t][1]:.3f}"]
        lines.append(",".join(cells))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"rl_{ds}.csv")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
