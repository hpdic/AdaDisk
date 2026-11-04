"""LID-scaled build search list (MCGI_LB_GAMMA) at fixed mean L_build.

Node i searches with L * exp(gamma * z_i) / mean during construction (z_i: standardized
LID), so the total build effort matches the uniform L. alpha is held at DiskANN's 1.2
(alpha_min = alpha_max = 1.2) to isolate the effect. Compared against stock DiskANN at
the same R and L (rl_sweep.py indices), on fixed and query-adaptive L.
Usage: python3 lb_sweep.py sift|glove|gist
"""
import concurrent.futures as cf
import os
import re
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rl_sweep  # noqa: E402
from m1_tune import BUILD, DS, ENV, paths  # noqa: E402

GAMMAS = [0.0, -0.5, 0.25, 0.5, 1.0]  # 0.0: uniform list, same batch, for build time and noise
RL = {"sift": [(32, 100), (64, 100)], "glove": [(32, 100), (64, 100)], "gist": [(32, 100), (48, 100)]}


def name(g, R, Lb):
    return f"lb{g}_R{R}_L{Lb}"


def build(ds, g, R, Lb, threads):
    d = f"{rl_sweep.OUT}/{ds}/{name(g, R, Lb)}"
    os.makedirs(d, exist_ok=True)
    p = paths(ds)
    base = f"{rl_sweep.OUT}/{ds}/diskann_R{R}_L{Lb}/idx"
    cmd = [BUILD, "--data_type", "float", "--dist_fn", "l2", "--data_path", p["base"],
           "--index_path_prefix", f"{d}/idx", "-R", str(R), "-L", str(Lb),
           "-B", rl_sweep.PQ_GB[ds], "-M", "4.0", "-T", str(threads),
           "--use_mcgi", "--lid_path", p["lid"], "--alpha_min", "1.2", "--alpha_max", "1.2",
           "--codebook_prefix", base]
    with open(f"{d}/build.log", "w") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                            env=dict(ENV, MCGI_LB_GAMMA=str(g))).returncode
    if rc == 0:
        for suf in ("_pq_pivots.bin", "_pq_compressed.bin"):
            shutil.copy(f"{base}{suf}", f"{d}/idx{suf}")
    return rc


def build_time(ds, n):
    with open(f"{rl_sweep.OUT}/{ds}/{n}/build.log", errors="replace") as f:
        m = re.findall(r"Indexing time: ([\d.]+)", f.read())
    return float(m[-1]) if m else float("nan")


def main():
    ds = sys.argv[1]
    jobs = [(g, R, Lb) for (R, Lb) in RL[ds] for g in GAMMAS]
    todo = [j for j in jobs if not os.path.exists(f"{rl_sweep.OUT}/{ds}/{name(*j)}/idx_disk.index")]
    # builds of a batch share the machine equally, so their build times are comparable
    with cf.ThreadPoolExecutor(max(1, len(todo))) as ex:
        rcs = list(ex.map(lambda j: build(ds, *j, max(16, 192 // max(1, len(todo)))), todo))
    for j, rc in zip(todo, rcs):
        if rc != 0:
            print(f"build failed: {j}", flush=True)
    names = [name(*j) for j in jobs if os.path.exists(f"{rl_sweep.OUT}/{ds}/{name(*j)}/idx_disk.index")]
    with cf.ThreadPoolExecutor(6) as ex:
        list(ex.map(lambda n: rl_sweep.search(ds, n), names))

    with open(paths(ds)["gt"], "rb") as f:
        nq, k = np.frombuffer(f.read(8), dtype=np.uint32)
        gt = np.frombuffer(f.read(int(nq) * int(k) * 4), dtype=np.uint32).reshape(nq, k)[:, :10]
    targets = DS[ds]["targets"]
    lines = ["config,build_s," + ",".join(f"io_fixed@{t:g},io_adapt@{t:g},gain_fixed@{t:g},gain_adapt@{t:g}"
                                          for t in targets)]
    for (R, Lb) in RL[ds]:
        ref_n = f"diskann_R{R}_L{Lb}"
        ref = rl_sweep.analyze(ds, ref_n, gt)
        for n in [ref_n, f"diskann_R{R}_L{2 * Lb}"] + [name(g, R, Lb) for g in GAMMAS]:
            if not os.path.exists(f"{rl_sweep.OUT}/{ds}/{n}/search.log"):
                continue
            r = rl_sweep.analyze(ds, n, gt)
            cells = [n, f"{build_time(ds, n):.0f}"]
            for t in targets:
                cells += [f"{r[t][0]:.1f}", f"{r[t][1]:.1f}", f"{ref[t][0] / r[t][0]:.3f}",
                          f"{ref[t][0] / r[t][1]:.3f}"]
            lines.append(",".join(cells))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"lb_{ds}.csv")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
