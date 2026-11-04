"""Clean build times of the three million-scale indexes of the paper (192 threads, one build at a time).

Same parameters as rl_sweep.py; the indexes go to ~/data/mcgi/buildtime and only the build logs are kept.
"""
import glob
import os
import subprocess
import time

from rl_sweep import BUILD, ENV, HOME, PQ_GB, paths

OUT = f"{HOME}/data/mcgi/buildtime"
CONFIGS = [("sift", 32, 50), ("glove", 64, 100), ("gist", 32, 100)]

for ds, R, Lb in CONFIGS:
    d = f"{OUT}/{ds}_R{R}_L{Lb}"
    os.makedirs(d, exist_ok=True)
    cmd = [BUILD, "--data_type", "float", "--dist_fn", "l2", "--data_path", paths(ds)["base"],
           "--index_path_prefix", f"{d}/idx", "-R", str(R), "-L", str(Lb),
           "-B", PQ_GB[ds], "-M", "4.0", "-T", "192"]
    t0 = time.time()
    with open(f"{d}/build.log", "w") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=ENV).returncode
    print(f"{ds} R{R} L{Lb}: rc {rc}, wall {time.time() - t0:.1f} s", flush=True)
    for f in glob.glob(f"{d}/idx*"):
        os.remove(f)
