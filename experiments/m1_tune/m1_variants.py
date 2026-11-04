"""Alpha-mapping variants of MCGI on SIFT1M / GloVe-100, in one-shot and sharded builds.

Variants are selected by environment variables read by the builder (see index.cpp):
MCGI_FINE=1 runs the last prune round at exactly alpha, MCGI_K sets the sigmoid
steepness, MCGI_DIR=-1 maps high LID to small alpha. Screening only: ranked by mean
I/Os per query at matched recall against the DiskANN reference of the same build mode
(one-shot: m1/<ds>/base, sharded: m1shard/<ds>/s1.2).
Usage: python3 m1_variants.py sift|glove
"""
import concurrent.futures as cf
import os
import shutil
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m1_shard  # noqa: E402
import m1_tune  # noqa: E402
from m1_tune import DS, at_recall, parse  # noqa: E402

VOUT = f"{m1_tune.HOME}/data/mcgi/m1var"
BUILD_PAR, SEARCH_PAR = 8, 6
F, K4, DN = {"MCGI_FINE": "1"}, {"MCGI_K": "4"}, {"MCGI_DIR": "-1"}
VARIANTS = {  # name: (alpha_min, alpha_max, env)
    "f_s1.3": (1.3, 1.3, F), "f_s1.6": (1.6, 1.6, F), "f_s2.0": (2.0, 2.0, F),
    "f_a1.2_2.0": (1.2, 2.0, F), "f_a1.2_3.0": (1.2, 3.0, F), "f_a1.3_3.0": (1.3, 3.0, F),
    "k4_a1.2_3.0": (1.2, 3.0, K4), "k4_a1.0_3.0": (1.0, 3.0, K4),
    "d_a1.2_3.0": (1.2, 3.0, DN), "d_a1.0_2.0": (1.0, 2.0, DN),
}
MODES = ["oneshot", "shard"]


def build(ds, mode, name):
    a, b, env = VARIANTS[name]
    run_env = dict(m1_tune.ENV, **env)
    if mode == "oneshot":
        os.makedirs(f"{VOUT}/{mode}/{ds}/{name}", exist_ok=True)
        base = f"{m1_tune.OUT}/{ds}/base/idx"
        prefix = f"{VOUT}/{mode}/{ds}/{name}/idx"
        cfg, p = DS[ds], m1_tune.paths(ds)
        cmd = [m1_tune.BUILD, "--data_type", "float", "--dist_fn", "l2", "--data_path", p["base"],
               "--index_path_prefix", prefix, "-R", str(cfg["R"]), "-L", str(cfg["L"]),
               "-B", "0.1", "-M", "1.0", "-T", "24", "--use_mcgi", "--lid_path", p["lid"],
               "--alpha_min", str(a), "--alpha_max", str(b), "--codebook_prefix", base]
        with open(f"{VOUT}/{mode}/{ds}/{name}/build.log", "w") as log:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=run_env).returncode
        if rc == 0:
            for suf in ("_pq_pivots.bin", "_pq_compressed.bin"):
                shutil.copy(f"{base}{suf}", f"{prefix}{suf}")
        return rc
    return m1_shard.build(ds, name, (a, b), 24, out=f"{VOUT}/{mode}", env=run_env)


def search(ds, mode, name):
    if name == "REF":
        d = f"{m1_tune.OUT}/{ds}/base" if mode == "oneshot" else f"{m1_shard.OUT}/{ds}/s1.2"
        # these reference logs are written by m1_tune / m1_shard; wait instead of racing them
        logs = [f"{d}/search_{t}.log" for t in ("screen", "rep0")]
        while not any(os.path.exists(lp) and len(parse(lp)) >= len(m1_shard.L_LIST) for lp in logs):
            time.sleep(30)
    else:
        d = f"{VOUT}/{mode}/{ds}/{name}"
    for tag in ("screen", "clean", "rep0"):
        lp = f"{d}/search_{tag}.log"
        if os.path.exists(lp) and len(parse(lp)) >= len(m1_shard.L_LIST):
            return np.array(parse(lp))
    p = m1_tune.paths(ds)
    cmd = [m1_shard.SEARCH, "--data_type", "float", "--dist_fn", "l2", "--index_path_prefix", f"{d}/idx",
           "--query_file", p["query"], "--gt_file", p["gt"], "-K", "10", "-L", *m1_shard.L_LIST,
           "--result_path", f"{d}/res_screen", "-T", str(m1_shard.SEARCH_T)]
    with open(f"{d}/search_screen.log", "w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=m1_tune.ENV)
    subprocess.run(f"rm -f {d}/res_screen*", shell=True)
    rows = parse(f"{d}/search_screen.log")
    return np.array(rows) if len(rows) == len(m1_shard.L_LIST) else None


def main():
    ds = sys.argv[1]
    jobs = [(m, n) for m in MODES for n in VARIANTS
            if not os.path.exists(f"{VOUT}/{m}/{ds}/{n}/idx_disk.index")]
    for i in range(0, len(jobs), BUILD_PAR):
        batch = jobs[i:i + BUILD_PAR]
        with cf.ThreadPoolExecutor(len(batch)) as ex:
            rcs = list(ex.map(lambda j: build(ds, *j), batch))
        for j, rc in zip(batch, rcs):
            if rc != 0:
                print(f"build failed: {j}", flush=True)
    print("builds done", flush=True)
    keys = [(m, "REF") for m in MODES] + [(m, n) for m in MODES for n in VARIANTS]
    with cf.ThreadPoolExecutor(SEARCH_PAR) as ex:
        curves = dict(zip(keys, ex.map(lambda k: search(ds, *k), keys)))
    targets = DS[ds]["targets"]
    lines = ["mode,config,alpha_min,alpha_max,env," + ",".join(f"io_gain@{t:g}" for t in targets)]
    for m, n in keys:
        c, ref = curves[(m, n)], curves[(m, "REF")]
        if c is None or n == "REF":
            continue
        a, b, env = VARIANTS[n]
        g = [at_recall(ref, t, 2) / at_recall(c, t, 2) for t in targets]
        lines.append(",".join([m, n, str(a), str(b), " ".join(f"{k}={v}" for k, v in env.items())]
                              + [f"{x:.3f}" for x in g]))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"variants_{ds}.csv")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
