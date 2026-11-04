"""In-memory reference points on GIST1M: HNSW (hnswlib) and IVF-Flat (faiss), 32 search threads.

build: build both indexes with all cores and save them (~/data/mcgi/memory_baselines)
run:   load each index and search the official queries once per setting; CSV with qps and Recall@10
Usage (~/venvs/baselines): python memory_gist.py build|run
"""
import os
import sys
import time

import faiss
import hnswlib
import numpy as np

HOME = os.path.expanduser("~")
DATA = f"{HOME}/hpdic/adadisk/experiments/data/gist"
OUT = f"{HOME}/data/mcgi/memory_baselines"
THREADS = 32
HNSW_M, HNSW_EFC = 16, 200
NLIST = 4096


def read_bin(path, dtype):
    with open(path, "rb") as f:
        n, d = np.frombuffer(f.read(8), dtype=np.uint32)
        return np.frombuffer(f.read(), dtype=dtype, count=int(n) * int(d)).reshape(int(n), int(d))


def recall(ids, gt):
    return 100.0 * np.mean([len(np.intersect1d(a, b)) / 10 for a, b in zip(ids[:, :10], gt[:, :10])])


def build():
    os.makedirs(OUT, exist_ok=True)
    x = np.ascontiguousarray(read_bin(f"{DATA}/gist_base.bin", np.float32))
    t = time.time()
    h = hnswlib.Index(space="l2", dim=x.shape[1])
    h.init_index(max_elements=len(x), M=HNSW_M, ef_construction=HNSW_EFC)
    h.add_items(x, num_threads=os.cpu_count())
    h.save_index(f"{OUT}/hnsw_M{HNSW_M}.bin")
    print(f"hnsw build {time.time() - t:.1f} s", flush=True)
    t = time.time()
    faiss.omp_set_num_threads(os.cpu_count())
    q = faiss.IndexFlatL2(x.shape[1])
    ivf = faiss.IndexIVFFlat(q, x.shape[1], NLIST)
    ivf.train(x)
    ivf.add(x)
    faiss.write_index(ivf, f"{OUT}/ivfflat_{NLIST}.index")
    print(f"ivf build {time.time() - t:.1f} s", flush=True)


def run():
    q = np.ascontiguousarray(read_bin(f"{DATA}/gist_query.bin", np.float32))
    gt = read_bin(f"{DATA}/gist_gt.bin", np.uint32)
    rows = ["method,param,qps,recall"]
    h = hnswlib.Index(space="l2", dim=q.shape[1])
    h.load_index(f"{OUT}/hnsw_M{HNSW_M}.bin")
    for ef in (10, 20, 40, 60, 80, 100, 150, 200, 300, 400, 600, 800):
        h.set_ef(ef)
        t = time.time()
        ids, _ = h.knn_query(q, k=10, num_threads=THREADS)
        rows.append(f"hnsw,{ef},{len(q) / (time.time() - t):.1f},{recall(ids, gt):.2f}")
        print(rows[-1], flush=True)
    faiss.omp_set_num_threads(THREADS)
    ivf = faiss.read_index(f"{OUT}/ivfflat_{NLIST}.index")
    for nprobe in (8, 16, 32, 64, 96, 128, 192, 256, 384, 512):
        ivf.nprobe = nprobe
        t = time.time()
        _, ids = ivf.search(q, 10)
        rows.append(f"ivfflat,{nprobe},{len(q) / (time.time() - t):.1f},{recall(ids, gt):.2f}")
        print(rows[-1], flush=True)
    open(f"{OUT}/memory_gist.csv", "w").write("\n".join(rows) + "\n")


if __name__ == "__main__":
    {"build": build, "run": run}[sys.argv[1]]()
