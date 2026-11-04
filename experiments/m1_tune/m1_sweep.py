"""Wide alpha_min x alpha_max sweep of offline MCGI on SIFT1M / GloVe-100.

Screening: builds run 8 at a time, searches 6 at a time; configs are ranked by
mean I/Os per query at matched recall, which concurrent runs do not distort.
Confirmation: the top configs and the DiskANN baseline are searched again one at a
time for clean QPS. Usage: python3 m1_sweep.py sift|glove
"""
import concurrent.futures as cf
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m1_tune import DS, ENV, L_LIST, OUT, SEARCH, SEARCH_T, at_recall, build, parse, paths  # noqa: E402

AMIN = [1.0, 1.1, 1.2, 1.3, 1.44, 1.6]
AMAX = [1.2, 1.5, 1.8, 2.0, 2.5, 3.0]
STATIC = [1.2, 1.5, 1.8, 2.0, 2.5, 3.0]
BUILD_PAR, SEARCH_PAR, TOP = 8, 6, 5
SUMMARY_DIR = os.path.dirname(os.path.abspath(__file__))


def configs():
    out = {}
    for a in AMIN:
        for b in AMAX:
            if b > a:
                out[f"a{a}_{b}"] = (a, b)
    for s in STATIC:
        out[f"s{s}"] = (s, s)
    return out


def search(ds, name, tag):
    p = paths(ds)
    prefix = f"{OUT}/{ds}/{name}/idx"
    log_path = f"{OUT}/{ds}/{name}/search_{tag}.log"
    if os.path.exists(log_path) and len(parse(log_path)) == len(L_LIST):
        return np.array(parse(log_path))
    cmd = [SEARCH, "--data_type", "float", "--dist_fn", "l2", "--index_path_prefix", prefix,
           "--query_file", p["query"], "--gt_file", p["gt"], "-K", "10", "-L", *L_LIST,
           "--result_path", f"{OUT}/{ds}/{name}/res_{tag}", "-T", str(SEARCH_T)]
    with open(log_path, "w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=ENV)
    subprocess.run(f"rm -f {OUT}/{ds}/{name}/res_{tag}*", shell=True)
    rows = parse(log_path)
    return np.array(rows) if len(rows) == len(L_LIST) else None


def row(ds, name, a, curve, base):
    cells = [name, f"{a[0]}", f"{a[1]}"]
    for t in DS[ds]["targets"]:
        io, bio = at_recall(curve, t, 2), at_recall(base, t, 2)
        q, bq = at_recall(curve, t, 1), at_recall(base, t, 1)
        cells += [f"{io:.1f}", f"{bio / io:.3f}", f"{q:.0f}", f"{q / bq:.3f}"]
    return cells


def main():
    ds = sys.argv[1]
    cfg = configs()
    if not os.path.exists(f"{OUT}/{ds}/base/idx_disk.index"):
        assert build(ds, "base", None, 192) == 0
    todo = [n for n in cfg if not os.path.exists(f"{OUT}/{ds}/{n}/idx_disk.index")]
    for i in range(0, len(todo), BUILD_PAR):
        batch = todo[i:i + BUILD_PAR]
        with cf.ThreadPoolExecutor(len(batch)) as ex:
            rcs = list(ex.map(lambda n: build(ds, n, cfg[n], 192 // len(batch)), batch))
        for n, rc in zip(batch, rcs):
            if rc != 0:
                print(f"build failed: {n}", flush=True)

    names = [n for n in cfg if os.path.exists(f"{OUT}/{ds}/{n}/idx_disk.index")]
    subprocess.run(["sudo", "-n", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"], check=True)
    with cf.ThreadPoolExecutor(SEARCH_PAR) as ex:
        curves = dict(zip(names + ["base"], ex.map(lambda n: search(ds, n, "screen"),
                                                    names + ["base"])))
    base = curves.pop("base")
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
    lines = [",".join(head + ["io_score"])]
    for n in ranked:
        if curves[n] is not None:
            lines.append(",".join(row(ds, n, cfg[n], curves[n], base) + [f"{io_score(n):.3f}"]))
    with open(f"{SUMMARY_DIR}/sweep_screen_{ds}.csv", "w") as f:
        f.write("\n".join(lines) + "\n")

    # confirmation: clean serial QPS for the baseline and the top configs
    lines = [",".join(head)]
    clean = {}
    for n in ["base"] + ranked[:TOP]:
        subprocess.run(["sudo", "-n", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"], check=True)
        clean[n] = search(ds, n, "clean")
    for n in ["base"] + ranked[:TOP]:
        a = cfg.get(n, ("stock", "1.2"))
        lines.append(",".join(row(ds, n, a, clean[n], clean["base"])))
    with open(f"{SUMMARY_DIR}/sweep_confirm_{ds}.csv", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
