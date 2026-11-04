"""Clean calibration cost of the three million-scale datasets: one feature-dump search of the
calibration queries over the calibration list sizes, alone on an idle machine with the page cache dropped.
Outputs go to ~/data/mcgi/calibtime; the fitted weights in final2 are not touched."""
import os
import subprocess

from final_eval import DATA, N0, RL, run_search
from m1_tune import parse

OUT = os.path.expanduser("~/data/mcgi/calibtime")
for ds, idx in [("sift", "diskann_R32_L50"), ("glove", "diskann_R64_L100"), ("gist", "diskann_R32_L100")]:
    out = f"{OUT}/{ds}"
    os.makedirs(out, exist_ok=True)
    subprocess.run(["sudo", "-n", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"], check=True)
    run_search(f"{RL}/{ds}/{idx}/idx", f"{DATA}/{ds}/{ds}_calib_query.bin", f"{DATA}/{ds}/{ds}_calib_gt.bin",
               out, {"MCGI_FEAT_DUMP": "1", "MCGI_FEAT_N0": str(N0)}, "calib").wait()
    subprocess.run(f"rm -f {out}/calib_*.bin", shell=True)
    r = parse(f"{out}/calib.log")
    print(ds, len(r), f"{sum(10000 / q for L, q, io, rec in r):.1f} s", flush=True)
