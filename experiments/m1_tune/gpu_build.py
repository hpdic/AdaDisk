"""GPU-assisted Vamana build (cuVS) written in DiskANN's format, for comparison with the CPU builder.

Builds the graph on one GPU with cuVS Vamana (R = graph_degree, L ~ visited_size, alpha 1.2),
saves it in DiskANN's in-memory graph format, lays it out with DiskANN's create_disk_layout,
and reuses the PQ codes of the CPU build with the same R and L (rl_sweep.py) so that only the
graph differs. The base vectors are read with ordinary buffered I/O into anonymous host memory
and copied to the GPU (no memory-mapped file is pinned or registered).
Usage (lidkit env): python gpu_build.py ds R visited_size [vamana_iters]
"""
import os
import shutil
import subprocess
import sys
import time

import cupy as cp
import numpy as np
from cuvs.neighbors import vamana

HOME = os.path.expanduser("~")
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
RL = f"{HOME}/data/mcgi/rl"
LAYOUT = f"{HOME}/hpdic/adadisk/build/apps/utils/create_disk_layout"


def main():
    ds, R, V = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    iters = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    base_bin = f"{DATA}/{ds}/{ds}_base.bin"
    with open(base_bin, "rb") as f:
        n, d = np.frombuffer(f.read(8), dtype=np.uint32)
        x = np.frombuffer(f.read(), dtype=np.float32).reshape(int(n), int(d))
    name = f"gpu_R{R}_V{V}_it{iters:g}"
    out = f"{RL}/{ds}/{name}"
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    xd = cp.asarray(x)
    params = vamana.IndexParams(metric="sqeuclidean", graph_degree=R, visited_size=V, vamana_iters=iters, alpha=1.2)
    index = vamana.build(params, xd)
    cp.cuda.Device().synchronize()
    t_build = time.time() - t0
    vamana.save(f"{out}/idx_mem.index", index, include_dataset=False)
    t1 = time.time()
    subprocess.run([LAYOUT, "float", base_bin, f"{out}/idx_mem.index", f"{out}/idx_disk.index"], check=True,
                   stdout=subprocess.DEVNULL)
    t_layout = time.time() - t1
    ref = f"{RL}/{ds}/diskann_R{R}_L50/idx" if os.path.exists(f"{RL}/{ds}/diskann_R{R}_L50") else None
    for suf in ("_pq_pivots.bin", "_pq_compressed.bin"):
        shutil.copy(f"{ref}{suf}", f"{out}/idx{suf}")
    os.remove(f"{out}/idx_mem.index")
    with open(f"{out}/build.log", "w") as f:
        f.write(f"gpu graph build s: {t_build:.1f}\ndisk layout s: {t_layout:.1f}\n")
    print(f"{ds} {name}: GPU graph {t_build:.1f} s, disk layout {t_layout:.1f} s")


if __name__ == "__main__":
    main()
