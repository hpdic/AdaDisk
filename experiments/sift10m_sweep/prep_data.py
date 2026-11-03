"""Prepare the SIFT10M subset of BIGANN in DiskANN .bin format.

base:  first 10M points of base.1B.u8bin
query: bigann_query.bvecs (10K)
gt:    official idx_10M.ivecs / dis_10M.fvecs, top-100
All I/O is buffered and sequential (no memory maps).
"""
import os
import sys
import numpy as np

SRC = os.path.expanduser("~/hpdic/datasets/bigann")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/data/mcgi/sift10m")
N = 10_000_000
D = 128
GT_K = 100
CHUNK = 1_000_000

os.makedirs(OUT, exist_ok=True)

# base
with open(f"{SRC}/base.1B.u8bin", "rb") as fin, open(f"{OUT}/base.u8bin", "wb") as fout:
    n, d = np.frombuffer(fin.read(8), dtype=np.uint32)
    assert d == D
    fout.write(np.array([N, D], dtype=np.uint32).tobytes())
    left = N
    while left:
        m = min(CHUNK, left)
        fout.write(fin.read(m * D))
        left -= m

# query (bvecs: int32 dim + dim uint8 per row)
raw = np.fromfile(f"{SRC}/bigann_query.bvecs", dtype=np.uint8)
q = raw.reshape(-1, 4 + D)
assert (q[:, :4].copy().view(np.int32) == D).all()
q = np.ascontiguousarray(q[:, 4:])
with open(f"{OUT}/query.u8bin", "wb") as f:
    f.write(np.array(q.shape, dtype=np.uint32).tobytes())
    f.write(q.tobytes())

# ground truth (ivecs / fvecs: int32 k + k values per row)
def read_vecs(path, dtype):
    a = np.fromfile(path, dtype=np.int32)
    k = a[0]
    a = a.reshape(-1, k + 1)
    return a[:, 1:].copy().view(dtype), k

ids, k = read_vecs(f"{SRC}/gnd/idx_10M.ivecs", np.int32)
dis, _ = read_vecs(f"{SRC}/gnd/dis_10M.fvecs", np.float32)
assert ids.shape[0] == q.shape[0] and k >= GT_K
ids = np.ascontiguousarray(ids[:, :GT_K]).astype(np.uint32)
dis = np.ascontiguousarray(dis[:, :GT_K])
assert ids.max() < N
with open(f"{OUT}/gt100.bin", "wb") as f:
    f.write(np.array(ids.shape, dtype=np.uint32).tobytes())
    f.write(ids.tobytes())
    f.write(dis.tobytes())

print(f"wrote {OUT}: base {N}x{D}, query {q.shape}, gt {ids.shape}")
