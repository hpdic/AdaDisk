"""Million-scale MCGI measurement and alpha-range grid (SIFT1M, GloVe-100).

Build settings follow scripts/run_exp.sh (offline MCGI with a precomputed LID file,
MCGI reuses the baseline PQ codebook). Each index is searched over the full L list
in one run with the page cache dropped first; one run per index.
Usage: python3 m1_tune.py sift|glove
"""
import concurrent.futures as cf
import os
import re
import shutil
import subprocess
import sys

import numpy as np

HOME = os.path.expanduser("~")
ADADISK = f"{HOME}/hpdic/adadisk"
BUILD = f"{ADADISK}/build/apps/build_disk_index"
SEARCH = f"{ADADISK}/build/apps/search_disk_index"
EXPDATA = f"{ADADISK}/experiments/data"
OUT = f"{HOME}/data/mcgi/m1"
ENV = dict(os.environ, TMPDIR=f"{HOME}/data/mcgi/tmp")
SUMMARY_DIR = os.path.dirname(os.path.abspath(__file__))

DS = {
    "sift": dict(R=32, L=50, targets=[95.0, 98.0, 99.0]),
    "glove": dict(R=32, L=100, targets=[80.0, 85.0, 87.0]),
    "gist": dict(R=32, L=100, targets=[85.0, 90.0, 95.0]),
}
L_LIST = "10 20 30 40 50 60 80 100 120 150 200 250 300 400".split()
SEARCH_T = 32
GRID = {
    "orig_sift": (1.0, 1.5),
    "orig_glove": (1.0, 1.1),
    "m_12_12": (1.2, 1.2),
    "m_12_15": (1.2, 1.5),
    "m_12_18": (1.2, 1.8),
    "m_12_20": (1.2, 2.0),
    "m_12_25": (1.2, 2.5),
    "m_144_20": (1.44, 2.0),
}
ROW = re.compile(r"^\s*(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$")


def paths(ds):
    d = f"{EXPDATA}/{ds}"
    return dict(base=f"{d}/{ds}_base.bin", query=f"{d}/{ds}_query.bin",
                gt=f"{d}/{ds}_gt.bin", lid=f"{d}/{ds}_lid.bin")


def build(ds, name, amin_amax, threads):
    cfg, p = DS[ds], paths(ds)
    os.makedirs(f"{OUT}/{ds}/{name}", exist_ok=True)
    prefix = f"{OUT}/{ds}/{name}/idx"
    base_prefix = f"{OUT}/{ds}/base/idx"
    cmd = [BUILD, "--data_type", "float", "--dist_fn", "l2", "--data_path", p["base"],
           "--index_path_prefix", prefix, "-R", str(cfg["R"]), "-L", str(cfg["L"]),
           "-B", "0.1", "-M", "1.0", "-T", str(threads)]
    if amin_amax is not None:
        cmd += ["--use_mcgi", "--lid_path", p["lid"], "--alpha_min", str(amin_amax[0]),
                "--alpha_max", str(amin_amax[1]), "--codebook_prefix", base_prefix]
    with open(f"{OUT}/{ds}/{name}/build.log", "w") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=ENV).returncode
    if rc == 0 and amin_amax is not None:
        for suf in ("_pq_pivots.bin", "_pq_compressed.bin"):
            shutil.copy(f"{base_prefix}{suf}", f"{prefix}{suf}")
    return rc


def parse(log_path):
    rows = []
    with open(log_path, errors="replace") as f:
        for line in f:
            m = ROW.match(line)
            if m:
                L, W, qps, lat, p999, ios, io_us, cpu, rec = map(float, m.groups())
                rows.append((L, qps, ios, rec))
    return rows


def curve(ds, name):
    """Search once over the full L list; reuse a complete earlier log."""
    log_path = f"{OUT}/{ds}/{name}/search_rep0.log"
    if os.path.exists(log_path) and len(parse(log_path)) == len(L_LIST):
        return np.array(parse(log_path))
    p = paths(ds)
    prefix = f"{OUT}/{ds}/{name}/idx"
    subprocess.run(["sudo", "-n", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"], check=True)
    cmd = [SEARCH, "--data_type", "float", "--dist_fn", "l2", "--index_path_prefix", prefix,
           "--query_file", p["query"], "--gt_file", p["gt"], "-K", "10", "-L", *L_LIST,
           "--result_path", f"{OUT}/{ds}/{name}/res", "-T", str(SEARCH_T)]
    with open(log_path, "w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=ENV, check=True)
    subprocess.run(f"rm -f {OUT}/{ds}/{name}/res*", shell=True)
    return np.array(parse(log_path))


def at_recall(mean, t, col):
    order = np.argsort(mean[:, 3])
    rec, val = mean[order, 3], mean[order, col]
    if t < rec[0] or t > rec[-1]:
        return float("nan")
    return float(np.interp(t, rec, val))


def main():
    ds = sys.argv[1]
    names = ["base"] + list(GRID)
    if not os.path.exists(f"{OUT}/{ds}/base/idx_disk.index"):
        assert build(ds, "base", None, 192) == 0
    todo = [n for n in GRID if not os.path.exists(f"{OUT}/{ds}/{n}/idx_disk.index")]
    if todo:
        with cf.ThreadPoolExecutor(len(todo)) as ex:
            rcs = list(ex.map(lambda n: build(ds, n, GRID[n], 192 // len(todo)), todo))
        assert all(rc == 0 for rc in rcs), list(zip(todo, rcs))

    targets = DS[ds]["targets"]
    lines = ["config,alpha_min,alpha_max," + ",".join(
        f"qps@{t:g},ios@{t:g},gain@{t:g}" for t in targets)]
    curves = {n: curve(ds, n) for n in names}
    base_mean = curves["base"]
    for n in names:
        mean = curves[n]
        a = GRID.get(n, ("stock", "1.2"))
        cells = [n, str(a[0]), str(a[1])]
        for t in targets:
            q, io, bq = at_recall(mean, t, 1), at_recall(mean, t, 2), at_recall(base_mean, t, 1)
            cells += [f"{q:.1f}", f"{io:.1f}", f"{q / bq:.3f}"]
        lines.append(",".join(cells))
        np.savetxt(f"{OUT}/{ds}/{n}/curve.csv", mean, delimiter=",",
                   header="L,qps,mean_ios,recall", comments="")
    with open(f"{SUMMARY_DIR}/summary_{ds}.csv", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
