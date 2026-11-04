"""Query-side ablation of the MCGI adaptive search list on the three million-scale indexes of the paper.

Variants (all fitted on the calibration queries, searched on the official queries of the same index):
  lid      only the two LID features (expanded-node LID, candidate-list LID)
  nolid    only the non-LID features (d1, d10, crowding)
  random   the paper's weights, but each query gets pmean + s * N(0, 1) with s = std of the calibration
           prediction (same spread of list sizes, assigned at random; MCGI_RANDOM_STD)
  g<x>     the paper's features with strength gamma = x instead of the calibrated one
  n<x>     the paper's features measured at checkpoint n0 = x instead of 24

prep: untimed (feature dump of the calibration queries at n0 = 12, fits) -> ~/data/mcgi/ablation/<ds>/variants.json
run:  timed, one search per variant, alone, page cache dropped -> ~/data/mcgi/ablation/<ds>/<variant>.log
Usage: python ablate.py prep|run
"""
import json
import os
import subprocess
import sys

import numpy as np

from final_eval import DATA, L_LIST, OUT as FINAL, RL, read_bin, run_search
from m1_tune import parse

HOME = os.path.expanduser("~")
OUT = f"{HOME}/data/mcgi/ablation"
SEARCH = f"{HOME}/hpdic/adadisk/build/apps/search_disk_index"
INDEXES = {"sift": "diskann_R32_L50", "glove": "diskann_R64_L100", "gist": "diskann_R32_L100"}
GAMMAS = (0.5, 0.7, 1.0, 1.3)
MAIN_FEATS = [0, 1, 2, 4, 5]


def calib(out, calib_gt):
    Ls = np.array([int(r[0]) for r in parse(f"{out}/calib.log")], dtype=float)
    with open(calib_gt, "rb") as f:
        n, k = np.frombuffer(f.read(8), dtype=np.uint32)
        gt = np.frombuffer(f.read(int(n) * int(k) * 4), dtype=np.uint32).reshape(n, k)[:, :10]
    rec = np.array([[len(np.intersect1d(a, b)) / 10 for a, b in
                     zip(read_bin(f"{out}/calib_{int(L)}_idx_uint32.bin", np.uint32)[:, :10], gt)] for L in Ls]).T
    st = [read_bin(f"{out}/calib_{int(L)}_stats.bin", np.float32) for L in Ls]
    io = np.array([s[:, 0] for s in st]).T
    return Ls, rec, io, st[-1]


def fit(Ls, rec, io, last, feats, stage, n0, gamma=None):
    """Same ridge fit and gamma selection as final_eval.fit, with the feature set, checkpoint stage, and
    (optionally) gamma as parameters. Returns the MCGI_ADAPT string and the std of the prediction."""
    F = last[:, [1 + 6 * stage + c for c in feats]].astype(np.float64)
    y = np.log([Ls[np.argmax(rec[i] >= rec[i].max())] for i in range(len(rec))])
    X = np.column_stack([F, np.ones(len(y))])
    w = np.linalg.solve(X.T @ X + 1e-3 * np.eye(X.shape[1]), X.T @ y)
    pred = X @ w
    fixed = np.array([(rec[:, i].mean() * 100, io[:, i].mean()) for i in range(len(Ls))])
    lo, hi = max(fixed[0, 0], 80.0), fixed[-1, 0]
    best = None
    for g in (GAMMAS if gamma is None else (gamma,)):
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
    full = np.zeros(6)
    full[feats] = w[:-1]
    adapt = ",".join(f"{x:.5f}" for x in full) + f",{w[-1]:.5f},{best[1]},{pred.mean():.5f},{n0},8"
    return adapt, float(pred.std()), best[1]


def prep():
    for ds, n in INDEXES.items():
        out = f"{OUT}/{ds}"
        os.makedirs(f"{out}/n12", exist_ok=True)
        cq, cg = f"{DATA}/{ds}/{ds}_calib_query.bin", f"{DATA}/{ds}/{ds}_calib_gt.bin"
        if not (os.path.exists(f"{out}/n12/calib.log") and len(parse(f"{out}/n12/calib.log")) == len(L_LIST)):
            run_search(f"{RL}/{ds}/{n}/idx", cq, cg, f"{out}/n12", {"MCGI_FEAT_DUMP": "1", "MCGI_FEAT_N0": "12"},
                       "calib").wait()
            subprocess.run(f"rm -f {out}/n12/calib_*_dists_float.bin", shell=True)
        main = calib(f"{FINAL}/{ds}/{n}", cg)
        paper = open(f"{FINAL}/{ds}/{n}/adapt.txt").read().split("MCGI_ADAPT=")[1].split()[0]
        a_main, s_main, g_main = fit(*main, MAIN_FEATS, 0, 24)
        assert a_main == paper, (ds, a_main, paper)  # the refit reproduces the paper's weights
        v = {"lid": {"MCGI_ADAPT": fit(*main, [0, 4], 0, 24)[0]},
             "nolid": {"MCGI_ADAPT": fit(*main, [1, 2, 5], 0, 24)[0]},
             "random": {"MCGI_ADAPT": paper, "MCGI_RANDOM_STD": f"{s_main:.5f}"},
             "n12": {"MCGI_ADAPT": fit(*calib(f"{out}/n12", cg), MAIN_FEATS, 0, 12)[0]},
             "n48": {"MCGI_ADAPT": fit(*main, MAIN_FEATS, 1, 48)[0]}}
        for g in GAMMAS:
            if g != g_main:
                v[f"g{g:g}"] = {"MCGI_ADAPT": fit(*main, MAIN_FEATS, 0, 24, gamma=g)[0]}
        json.dump({"gamma": g_main, "variants": v}, open(f"{out}/variants.json", "w"), indent=1)
        print(ds, "gamma", g_main, "pred std", round(s_main, 4), list(v), flush=True)


def run():
    for ds, n in INDEXES.items():
        out = f"{OUT}/{ds}"
        v = json.load(open(f"{out}/variants.json"))["variants"]
        for tag, env in v.items():
            if os.path.exists(f"{out}/{tag}.log") and len(parse(f"{out}/{tag}.log")) == len(L_LIST):
                continue
            subprocess.run(["sudo", "-n", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"], check=True)
            e = {k: x for k, x in os.environ.items() if not k.startswith("MCGI_")}
            e.update(env, TMPDIR=f"{HOME}/data/mcgi/tmp")
            cmd = [SEARCH, "--data_type", "float", "--dist_fn", "l2", "--index_path_prefix", f"{RL}/{ds}/{n}/idx",
                   "--query_file", f"{DATA}/{ds}/{ds}_query.bin", "--gt_file", f"{DATA}/{ds}/{ds}_gt.bin",
                   "-K", "10", "-L", *L_LIST, "-T", "32", "--result_path", f"{out}/res"]
            with open(f"{out}/{tag}.log", "w") as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=e, check=True)
            subprocess.run(f"rm -f {out}/res_*", shell=True)
            print(ds, tag, "done", flush=True)


if __name__ == "__main__":
    {"prep": prep, "run": run}[sys.argv[1]]()
