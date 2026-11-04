"""Convert a DiskANN PQ pivots file (offsets: pivots, centroid, chunk offsets, end) to the PipeANN layout
(offsets: pivots, centroid, identity rearrangement, chunk offsets, end). Usage: src dst"""
import sys

import numpy as np


def read_block(buf, off, dtype):
    nr, nc = np.frombuffer(buf, dtype=np.int32, count=2, offset=off)
    return np.frombuffer(buf, dtype=dtype, count=int(nr) * int(nc), offset=off + 8).reshape(int(nr), int(nc))


def block(a):
    a = np.ascontiguousarray(a)
    return np.array(a.shape, dtype=np.int32).tobytes() + a.tobytes()


buf = open(sys.argv[1], "rb").read()
offs = read_block(buf, 0, np.uint64).ravel()
assert len(offs) == 4, offs
tables = read_block(buf, int(offs[0]), np.float32)
centroid = read_block(buf, int(offs[1]), np.float32)
chunks = read_block(buf, int(offs[2]), np.uint32)
dim = tables.shape[1]
parts = [block(tables), block(centroid), block(np.arange(dim, dtype=np.uint32).reshape(dim, 1)), block(chunks)]
new = [4096]
for p in parts:
    new.append(new[-1] + len(p))
out = bytearray(new[0])
head = block(np.array(new, dtype=np.uint64).reshape(-1, 1))
out[:len(head)] = head
for p in parts:
    out += p
open(sys.argv[2], "wb").write(bytes(out))
print(sys.argv[2], tables.shape, chunks.shape)
