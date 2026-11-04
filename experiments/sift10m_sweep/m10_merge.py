"""Sharded builds on SIFT10M: DiskANN's random-truncation merge vs alpha-pruned merge.

All indices are built through the AMCGI path with a 1 GB memory budget (sharded build,
as at billion scale) and share the PQ codes of the stock build base12. st_s1.2 is stock
DiskANN (alpha = 1.2, random truncation at merge); mp_* set MCGI_MERGE_PRUNE=1 so the
union of shard neighborhoods is pruned back to R with the node's alpha.
Screening by mean I/Os per query at matched recall against st_s1.2.
Usage: python3 m10_merge.py [data_dir memory_gb pq_prefix]   (default: SIFT10M, 1 GB, base12)
"""
import concurrent.futures as cf
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_results import interp, read_rows  # noqa: E402
from tune import BUILD, ENV, SEARCH  # noqa: E402
import tune  # noqa: E402

DATA, M, STOCK = (sys.argv[1], float(sys.argv[2]), sys.argv[3]) if len(sys.argv) > 3 else (tune.DATA, 1, tune.STOCK)
OUT = f"{DATA}/merge"
R, L, B = 32, 50, 2
LID = (19.5, 7.9)
L_LIST = "10 20 30 40 50 60 80 100 150 200".split()
TARGETS = [90.0, 95.0, 98.0]
MP = {"MCGI_MERGE_PRUNE": "1"}
CONFIGS = {
    "st_s1.2": (1.2, 1.2, {}), "st_a1.2_2.0": (1.2, 2.0, {}),
    "mp_s1.2": (1.2, 1.2, MP), "mp_s1.5": (1.5, 1.5, MP), "mp_s2.0": (2.0, 2.0, MP),
    "mp_a1.2_2.0": (1.2, 2.0, MP), "mp_a1.0_2.0": (1.0, 2.0, MP), "mp_a1.2_3.0": (1.2, 3.0, MP),
}


def build(name, threads):
    a, b, env = CONFIGS[name]
    d = f"{OUT}/{name}"
    os.makedirs(d, exist_ok=True)
    prefix = f"{d}/idx"
    cmd = [BUILD, "--data_type", "uint8", "--dist_fn", "l2", "--data_path", f"{DATA}/base.u8bin",
           "--index_path_prefix", prefix, "-R", str(R), "-L", str(L), "-B", str(B), "-M", str(M),
           "-T", str(threads), "--use_amcgi", "--alpha_min", str(a), "--alpha_max", str(b),
           "--lid_avg", str(LID[0]), "--lid_std", str(LID[1])]
    with open(f"{d}/build.log", "w") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=dict(ENV, **env)).returncode
    if rc != 0:
        return rc
    for suf in ("_pq_pivots.bin", "_pq_compressed.bin"):
        if not os.path.exists(f"{prefix}{suf}"):
            os.symlink(f"{STOCK}{suf}", f"{prefix}{suf}")
    for suf in ("_medoids.bin", "_centroids.bin"):
        if os.path.exists(f"{prefix}{suf}"):
            os.rename(f"{prefix}{suf}", f"{prefix}_disk.index{suf}")
    if os.path.exists(f"{prefix}_mem.index"):
        os.remove(f"{prefix}_mem.index")
    return 0


def search(name):
    d = f"{OUT}/{name}"
    log_path = f"{d}/search_screen.log"
    if os.path.exists(log_path) and len(read_rows(log_path)) == len(L_LIST):
        return read_rows(log_path)
    cmd = [SEARCH, "--data_type", "uint8", "--dist_fn", "l2", "--index_path_prefix", f"{d}/idx",
           "--query_file", f"{DATA}/query.u8bin", "--gt_file", f"{DATA}/gt100.bin",
           "-K", "10", "-L", *L_LIST, "-W", "2", "-T", "32", "--result_path", f"{d}/res"]
    with open(log_path, "w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=ENV)
    subprocess.run(f"rm -f {d}/res*", shell=True)
    return read_rows(log_path)


def main():
    todo = [n for n in CONFIGS if not os.path.exists(f"{OUT}/{n}/idx_disk.index")]
    with cf.ThreadPoolExecutor(max(1, len(todo))) as ex:
        rcs = list(ex.map(lambda n: build(n, 192 // max(1, len(todo))), todo))
    for n, rc in zip(todo, rcs):
        if rc != 0:
            print(f"build failed: {n}", flush=True)
    print("builds done", flush=True)
    names = [n for n in CONFIGS if os.path.exists(f"{OUT}/{n}/idx_disk.index")]
    with cf.ThreadPoolExecutor(4) as ex:
        rows = dict(zip(names, ex.map(search, names)))
    ref = rows["st_s1.2"]
    lines = ["config,alpha_min,alpha_max,env," + ",".join(f"ios@{t:g},io_gain@{t:g}" for t in TARGETS)]
    for n in names:
        a, b, env = CONFIGS[n]
        cells = [n, str(a), str(b), " ".join(f"{k}={v}" for k, v in env.items())]
        for t in TARGETS:
            io, rio = interp(rows[n], t, 2), interp(ref, t, 2)
            cells += [f"{io:.1f}", f"{rio / io:.3f}"]
        lines.append(",".join(cells))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"merge_{os.path.basename(DATA)}.csv")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
