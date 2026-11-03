"""Hyperparameter search for online MCGI on SIFT10M (first 10M points of BIGANN).

Every index, including the DiskANN reference, is built through the same AMCGI code
path. static12 (alpha_min = alpha_max = 1.2) is stock DiskANN's alpha = 1.2. All
indices share the PQ codes of the stock build base12 (PQ depends only on the data).

Phase 1 runs a grid in two build modes: shard (M = 1 GB, sharded build as at 1B)
and oneshot (M = 64 GB, single in-memory build). Phase 2 runs a coordinate search
over alpha_min, alpha_max, lid_avg, lid_std in the mode where MCGI scored best.
Score: geometric mean of the QPS gain over static12 at Recall@10 90 / 95 / 98.
Builds of a round run in parallel; searches then run one at a time with no build
running, so that builds do not disturb the I/O measurement.
"""
import concurrent.futures as cf
import csv
import math
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_results import interp, read_rows  # noqa: E402

HOME = os.path.expanduser("~")
ADADISK = f"{HOME}/hpdic/adadisk"
BUILD = f"{ADADISK}/build/apps/build_disk_index"
SEARCH = f"{ADADISK}/build/apps/search_disk_index"
DATA = f"{HOME}/data/mcgi/sift10m"
IDX = f"{DATA}/indices"
LOGS = f"{DATA}/logs"
HIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tune_history.csv")
ENV = dict(os.environ, TMPDIR=f"{HOME}/data/mcgi/tmp")
STOCK = f"{IDX}/base12_R32_L50"  # stock DiskANN build; source of the shared PQ files

R, L, B = 32, 50, 2
MODE_M = {"shard": 1, "oneshot": 64}
SEARCH_T = 96
L_LIST = "10 20 30 40 50 60 80 100 120 150 200".split()
TARGETS = [90.0, 95.0, 98.0]
MAX_PAR = 8
MAX_ROUNDS = 10
LID0 = (19.5, 7.9)

GRID = {
    "static12": (1.2, 1.2),
    "mcgi_10_15": (1.0, 1.5),
    "mcgi_12_18": (1.2, 1.8),
    "mcgi_12_20": (1.2, 2.0),
    "mcgi_12_25": (1.2, 2.5),
    "mcgi_144_20": (1.44, 2.0),
    "static15": (1.5, 1.5),
}

KEYS = ["amin", "amax", "avg", "std"]
STEP0 = {"amin": 0.1, "amax": 0.2, "avg": 2.0, "std": 2.0}
STEP_MIN = {"amin": 0.025, "amax": 0.05, "avg": 0.5, "std": 0.5}
BOUNDS = {"amin": (1.0, 2.0), "amax": (1.0, 3.0), "avg": (5.0, 40.0), "std": (1.0, 20.0)}


def log_rows(name):
    path = f"{LOGS}/{name}_search.log"
    return read_rows(path) if os.path.exists(path) else None


def gain_score(name, ref):
    rows, base = log_rows(name), log_rows(ref)
    if rows is None:
        return None
    if not rows or not base:
        return float("-inf")
    gains = []
    for t in TARGETS:
        q, b = interp(rows, t, 1), interp(base, t, 1)
        if math.isnan(q) or math.isnan(b):
            return float("-inf")
        gains.append(q / b)
    return math.exp(sum(map(math.log, gains)) / len(gains))


def build(name, p, mode, threads):
    prefix = f"{IDX}/{name}"
    cmd = [BUILD, "--data_type", "uint8", "--dist_fn", "l2",
           "--data_path", f"{DATA}/base.u8bin", "--index_path_prefix", prefix,
           "-R", str(R), "-L", str(L), "-B", str(B), "-M", str(MODE_M[mode]), "-T", str(threads),
           "--use_amcgi", "--alpha_min", f"{p[0]}", "--alpha_max", f"{p[1]}",
           "--lid_avg", f"{p[2]}", "--lid_std", f"{p[3]}"]
    with open(f"{LOGS}/{name}_build.log", "w") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=ENV).returncode
    if rc != 0:
        return rc
    # the AMCGI path writes neither PQ files nor the medoid names search_disk_index loads
    for suf in ("_pq_pivots.bin", "_pq_compressed.bin"):
        os.symlink(f"{STOCK}{suf}", f"{prefix}{suf}")
    for suf in ("_medoids.bin", "_centroids.bin"):
        if os.path.exists(f"{prefix}{suf}"):
            os.rename(f"{prefix}{suf}", f"{prefix}_disk.index{suf}")
    os.remove(f"{prefix}_mem.index")
    return 0


def search(name):
    prefix = f"{IDX}/{name}"
    subprocess.run(["sudo", "-n", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"], check=True)
    cmd = [SEARCH, "--data_type", "uint8", "--dist_fn", "l2", "--index_path_prefix", prefix,
           "--query_file", f"{DATA}/query.u8bin", "--gt_file", f"{DATA}/gt100.bin",
           "-K", "10", "-L", *L_LIST, "-W", "2", "--num_nodes_to_cache", "10000",
           "-T", str(SEARCH_T), "--result_path", f"{IDX}/{name}_res"]
    with open(f"{LOGS}/{name}_search.log", "w") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=ENV).returncode
    subprocess.run(f"rm -f {prefix}_*", shell=True)
    return rc


def run_batch(jobs):
    """jobs: list of (name, params, mode); builds in parallel, then searches serially."""
    todo = [j for j in jobs if log_rows(j[0]) is None]
    if not todo:
        return
    threads = max(16, 192 // len(todo))
    with cf.ThreadPoolExecutor(len(todo)) as ex:
        rcs = list(ex.map(lambda j: build(j[0], j[1], j[2], threads), todo))
    for (name, _, _), rc in zip(todo, rcs):
        if rc == 0:
            search(name)
        else:
            print(f"build failed: {name}", flush=True)


def record(rnd, mode, name, p, s):
    new = not os.path.exists(HIST)
    with open(HIST, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["round", "mode", "config", "alpha_min", "alpha_max", "lid_avg", "lid_std", "score"])
        w.writerow([rnd, mode, name, *p, f"{s:.4f}"])


def tname(mode, p):
    return "{}_t_{:.3f}_{:.3f}_c{:.1f}_s{:.1f}".format(mode, *p)


def neighbors(best, step):
    cands = []
    for i, k in enumerate(KEYS):
        for sign in (1, -1):
            p = list(best)
            lo, hi = BOUNDS[k]
            p[i] = round(min(hi, max(lo, p[i] + sign * step[k])), 3)
            if p[1] < p[0]:
                continue
            p = tuple(p)
            if p != tuple(best) and p not in cands:
                cands.append(p)
    return cands[:MAX_PAR]


def main():
    # phase 1: grid in both modes; static12 twice in shard mode to measure build noise
    jobs = []
    for mode in MODE_M:
        for tag, (amin, amax) in GRID.items():
            jobs.append((f"{mode}_{tag}", (amin, amax, *LID0), mode))
    jobs.append(("shard_static12_rep", (1.2, 1.2, *LID0), "shard"))
    for i in range(0, len(jobs), MAX_PAR):
        run_batch(jobs[i:i + MAX_PAR])

    best_mode, best_s, best = None, float("-inf"), None
    for name, p, mode in jobs:
        s = gain_score(name, f"{mode}_static12")
        s = float("-inf") if s is None else s
        record(0, mode, name, p, s)
        print(f"grid {name}: {s:.4f}", flush=True)
        if name.endswith("static12") or name.endswith("_rep"):
            continue
        if s > best_s:
            best_mode, best_s, best = mode, s, p
    print(f"grid best {best_mode} {best} score {best_s:.4f}", flush=True)

    # phase 2: coordinate search in the winning mode
    ref = f"{best_mode}_static12"
    step = dict(STEP0)
    for rnd in range(1, MAX_ROUNDS + 1):
        cands = neighbors(best, step)
        run_batch([(tname(best_mode, p), p, best_mode) for p in cands])
        res = []
        for p in cands:
            s = gain_score(tname(best_mode, p), ref)
            s = float("-inf") if s is None else s
            record(rnd, best_mode, tname(best_mode, p), p, s)
            res.append((s, p))
        top_s, top = max(res)
        print(f"round {rnd}: best candidate {top} score {top_s:.4f} (incumbent {best_s:.4f})", flush=True)
        if top_s > best_s:
            best_s, best = top_s, top
        else:
            step = {k: v / 2 for k, v in step.items()}
            if all(step[k] < STEP_MIN[k] for k in KEYS):
                break
    print(f"final best {best_mode} {best} score {best_s:.4f}", flush=True)


if __name__ == "__main__":
    main()
