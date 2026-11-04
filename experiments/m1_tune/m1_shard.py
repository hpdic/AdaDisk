"""Sharded-build alpha sweep of online MCGI (AMCGI) on SIFT1M / GloVe-100.

A small memory budget (-M) forces the sharded build used at billion scale: shards are
built at degree 2R/3 without saturation and merged, so alpha sets the node degree.
Every index, including the DiskANN reference s1.2 (alpha_min = alpha_max = 1.2),
goes through the same AMCGI path and shares the PQ codes of the one-shot base build.
Builds run while other searches hold the disk; searches wait for a given PID to exit.
Usage: python3 m1_shard.py sift|glove [wait_pid]
"""
import concurrent.futures as cf
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m1_tune import BUILD, DS, ENV, HOME, at_recall, paths  # noqa: E402
from m1_sweep import row  # noqa: E402

OUT = f"{HOME}/data/mcgi/m1shard"
PQ_FROM = f"{HOME}/data/mcgi/m1"
SEARCH = f"{HOME}/hpdic/adadisk/build/apps/search_disk_index"
L_LIST = "10 20 30 40 50 60 80 100 150 200 300".split()
SEARCH_T = 32
MEM_GB = 0.1
LID = {"sift": (16.1, 9.2), "glove": (35.3, 18.1)}  # mean, std of the offline LID files
AMIN = [1.0, 1.2]
AMAX = [1.5, 2.0, 2.5, 3.0]
STATIC = [1.2, 1.5, 2.0, 2.5, 3.0]
BUILD_PAR, SEARCH_PAR, TOP = 6, 6, 5
SUMMARY_DIR = os.path.dirname(os.path.abspath(__file__))


def configs():
    out = {f"a{a}_{b}": (a, b) for a in AMIN for b in AMAX}
    out.update({f"s{s}": (s, s) for s in STATIC})
    return out


def build(ds, name, ab, threads, out=None, env=None):
    cfg, p = DS[ds], paths(ds)
    d = f"{out or OUT}/{ds}/{name}"
    os.makedirs(d, exist_ok=True)
    prefix = f"{d}/idx"
    cmd = [BUILD, "--data_type", "float", "--dist_fn", "l2", "--data_path", p["base"],
           "--index_path_prefix", prefix, "-R", str(cfg["R"]), "-L", str(cfg["L"]),
           "-B", "0.1", "-M", str(MEM_GB), "-T", str(threads), "--use_amcgi",
           "--alpha_min", str(ab[0]), "--alpha_max", str(ab[1]),
           "--lid_avg", str(LID[ds][0]), "--lid_std", str(LID[ds][1])]
    with open(f"{d}/build.log", "w") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env or ENV).returncode
    if rc != 0:
        return rc
    # the AMCGI path writes neither PQ files nor the medoid names search_disk_index loads
    for suf in ("_pq_pivots.bin", "_pq_compressed.bin"):
        if not os.path.exists(f"{prefix}{suf}"):
            os.symlink(f"{PQ_FROM}/{ds}/base/idx{suf}", f"{prefix}{suf}")
    for suf in ("_medoids.bin", "_centroids.bin"):
        if os.path.exists(f"{prefix}{suf}"):
            os.rename(f"{prefix}{suf}", f"{prefix}_disk.index{suf}")
    if os.path.exists(f"{prefix}_mem.index"):
        os.remove(f"{prefix}_mem.index")
    return 0


def parse(path):
    from m1_tune import parse as p
    return p(path)


def search(ds, name, tag):
    p = paths(ds)
    d = f"{OUT}/{ds}/{name}"
    log_path = f"{d}/search_{tag}.log"
    if os.path.exists(log_path) and len(parse(log_path)) == len(L_LIST):
        return np.array(parse(log_path))
    cmd = [SEARCH, "--data_type", "float", "--dist_fn", "l2", "--index_path_prefix", f"{d}/idx",
           "--query_file", p["query"], "--gt_file", p["gt"], "-K", "10", "-L", *L_LIST,
           "--result_path", f"{d}/res_{tag}", "-T", str(SEARCH_T)]
    with open(log_path, "w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=ENV)
    subprocess.run(f"rm -f {d}/res_{tag}*", shell=True)
    rows = parse(log_path)
    return np.array(rows) if len(rows) == len(L_LIST) else None


def degree(ds, name):
    with open(f"{OUT}/{ds}/{name}/build.log", errors="replace") as f:
        lines = [l for l in f if "Index built with degree" in l]
    return np.mean([float(l.split("avg:")[1].split()[0]) for l in lines]) if lines else float("nan")


def drop_caches():
    subprocess.run(["sudo", "-n", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"], check=True)


def main():
    ds = sys.argv[1]
    wait_pid = int(sys.argv[2]) if len(sys.argv) > 2 else None
    cfg = configs()
    todo = [n for n in cfg if not os.path.exists(f"{OUT}/{ds}/{n}/idx_disk.index")]
    for i in range(0, len(todo), BUILD_PAR):
        batch = todo[i:i + BUILD_PAR]
        with cf.ThreadPoolExecutor(len(batch)) as ex:
            rcs = list(ex.map(lambda n: build(ds, n, cfg[n], 192 // len(batch)), batch))
        for n, rc in zip(batch, rcs):
            if rc != 0:
                print(f"build failed: {n}", flush=True)
    print("builds done", flush=True)
    while wait_pid and os.path.exists(f"/proc/{wait_pid}"):
        time.sleep(30)

    names = [n for n in cfg if os.path.exists(f"{OUT}/{ds}/{n}/idx_disk.index")]
    drop_caches()
    with cf.ThreadPoolExecutor(SEARCH_PAR) as ex:
        curves = dict(zip(names, ex.map(lambda n: search(ds, n, "screen"), names)))
    base = curves["s1.2"]
    targets = DS[ds]["targets"]
    head = ["config", "alpha_min", "alpha_max"]
    for t in targets:
        head += [f"ios@{t:g}", f"io_gain@{t:g}", f"qps@{t:g}", f"qps_gain@{t:g}"]

    def io_score(n):
        c = curves[n]
        if c is None:
            return float("-inf")
        g = [at_recall(base, t, 2) / at_recall(c, t, 2) for t in targets]
        return float(np.exp(np.mean(np.log(g)))) if not np.isnan(g).any() else float("-inf")

    ranked = sorted(names, key=io_score, reverse=True)
    lines = [",".join(head + ["avg_degree", "io_score"])]
    for n in ranked:
        if curves[n] is not None:
            lines.append(",".join(row(ds, n, cfg[n], curves[n], base)
                                  + [f"{degree(ds, n):.2f}", f"{io_score(n):.3f}"]))
    with open(f"{SUMMARY_DIR}/shard_screen_{ds}.csv", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)

    # confirmation: clean serial QPS for the reference and the top configs
    lines = [",".join(head)]
    clean = {}
    order = ["s1.2"] + [n for n in ranked[:TOP] if n != "s1.2"]
    for n in order:
        drop_caches()
        clean[n] = search(ds, n, "clean")
    for n in order:
        lines.append(",".join(row(ds, n, cfg[n], clean[n], clean["s1.2"])))
    with open(f"{SUMMARY_DIR}/shard_confirm_{ds}.csv", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
