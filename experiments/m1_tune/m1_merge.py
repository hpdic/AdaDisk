"""Sharded build with LID-adaptive merge pruning (MCGI_MERGE_PRUNE=1) on SIFT1M / GloVe-100.

DiskANN merges shard graphs by taking the union of a node's shard neighborhoods and
truncating it at random to R. With MCGI_MERGE_PRUNE=1 the union is pruned back to R
with the node's alpha instead. Screening by mean I/Os against the stock sharded
DiskANN build m1shard/<ds>/s1.2 (random truncation, alpha = 1.2).
Usage: python3 m1_merge.py sift|glove
"""
import concurrent.futures as cf
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m1_shard  # noqa: E402
import m1_tune  # noqa: E402
from m1_tune import DS, at_recall, parse  # noqa: E402
from m1_variants import search as vsearch  # noqa: E402
import m1_variants  # noqa: E402

MOUT = f"{m1_tune.HOME}/data/mcgi/m1var/merge"
MP = {"MCGI_MERGE_PRUNE": "1"}
VARIANTS = {
    "mp_s1.2": (1.2, 1.2, MP), "mp_s1.5": (1.5, 1.5, MP), "mp_s2.0": (2.0, 2.0, MP),
    "mp_s3.0": (3.0, 3.0, MP),
    "mp_a1.0_2.0": (1.0, 2.0, MP), "mp_a1.2_2.0": (1.2, 2.0, MP), "mp_a1.2_3.0": (1.2, 3.0, MP),
    "mp_a1.0_3.0": (1.0, 3.0, MP), "mp_d_a1.2_3.0": (1.2, 3.0, dict(MP, MCGI_DIR="-1")),
    "mp_nofill_a1.2_3.0": (1.2, 3.0, dict(MP, MCGI_MERGE_FILL="0")),
}


def build(ds, name):
    a, b, env = VARIANTS[name]
    return m1_shard.build(ds, name, (a, b), 24, out=MOUT, env=dict(m1_tune.ENV, **env))


def main():
    ds = sys.argv[1]
    todo = [n for n in VARIANTS if not os.path.exists(f"{MOUT}/{ds}/{n}/idx_disk.index")]
    with cf.ThreadPoolExecutor(8) as ex:
        rcs = list(ex.map(lambda n: build(ds, n), todo))
    for n, rc in zip(todo, rcs):
        if rc != 0:
            print(f"build failed: {n}", flush=True)
    print("builds done", flush=True)
    m1_variants.VOUT = os.path.dirname(MOUT)  # vsearch reads <VOUT>/<mode>/<ds>/<name>
    keys = [("shard", "REF")] + [("merge", n) for n in VARIANTS]
    with cf.ThreadPoolExecutor(6) as ex:
        curves = dict(zip(keys, ex.map(lambda k: vsearch(ds, *k), keys)))
    ref = curves[("shard", "REF")]
    targets = DS[ds]["targets"]
    lines = ["config,alpha_min,alpha_max,env," + ",".join(f"io_gain@{t:g}" for t in targets)]
    for (_, n), c in curves.items():
        if n == "REF" or c is None:
            continue
        a, b, env = VARIANTS[n]
        g = [at_recall(ref, t, 2) / at_recall(c, t, 2) for t in targets]
        lines.append(",".join([n, str(a), str(b), " ".join(f"{k}={v}" for k, v in env.items())]
                              + [f"{x:.3f}" for x in g]))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"merge_{ds}.csv")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
