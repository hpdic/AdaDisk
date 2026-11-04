"""Calibration query sets for the query-adaptive L, disjoint from the official queries.

SIFT1M and GIST1M: the first 10K vectors of the dataset's learn set. GloVe-100 has no learn
set: 10K base vectors drawn with a fixed seed, each excluded from its own ground truth.
Ground truth: exact top-100 by squared L2, chunked matrix products in RAM (buffered reads,
no memory maps). Writes <ds>_calib_query.bin and <ds>_calib_gt.bin (ids, then distances).
Usage: OPENBLAS_NUM_THREADS=32 python3 calib_prep.py sift|glove|gist
"""
import os
import sys

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
NQ, K = 10000, 100


def read_bin(path):
    with open(path, "rb") as f:
        n, d = np.frombuffer(f.read(8), dtype=np.uint32)
        return np.frombuffer(f.read(), dtype=np.float32).reshape(int(n), int(d))


def read_fvecs(path, n):
    with open(path, "rb") as f:
        d = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
        f.seek(0)
        a = np.frombuffer(f.read(n * (d + 1) * 4), dtype=np.float32).reshape(n, d + 1)
    return np.ascontiguousarray(a[:, 1:])


def write_bin(path, *arrs):
    with open(path, "wb") as f:
        f.write(np.array(arrs[0].shape, dtype=np.uint32).tobytes())
        for a in arrs:
            f.write(np.ascontiguousarray(a).tobytes())


def main():
    ds = sys.argv[1]
    d = f"{DATA}/{ds}"
    base = read_bin(f"{d}/{ds}_base.bin")
    self_ids = None
    if ds == "glove":
        self_ids = np.sort(np.random.default_rng(0).choice(len(base), NQ, replace=False))
        q = base[self_ids].copy()
    else:
        q = read_fvecs(f"{d}/{ds}_learn.fvecs", NQ)
    bn = (base.astype(np.float64) ** 2).sum(1).astype(np.float32)
    ids = np.empty((NQ, K), dtype=np.uint32)
    dis = np.empty((NQ, K), dtype=np.float32)
    for s in range(0, NQ, 500):
        qq = q[s:s + 500]
        dd = bn[None, :] - 2 * qq @ base.T + (qq.astype(np.float64) ** 2).sum(1, keepdims=True).astype(np.float32)
        if self_ids is not None:
            dd[np.arange(len(qq)), self_ids[s:s + 500]] = np.inf
        part = np.argpartition(dd, K, axis=1)[:, :K]
        pd = np.take_along_axis(dd, part, 1)
        o = np.argsort(pd, axis=1)
        ids[s:s + 500] = np.take_along_axis(part, o, 1)
        dis[s:s + 500] = np.maximum(np.take_along_axis(pd, o, 1), 0)
    write_bin(f"{d}/{ds}_calib_query.bin", q)
    write_bin(f"{d}/{ds}_calib_gt.bin", ids, dis)
    print(f"{ds}: calib queries {q.shape}, gt {ids.shape}")


if __name__ == "__main__":
    main()
