"""Export the comparison, ablation, and latency runs into the CSV files of the paper (paper/NeurIPS26/script/data).

gist_compare.csv  R,method,L,qps,mean_ios,recall   DiskANN / MCGI (our engine) and PipeANN (I/O width 4 and 32)
                                                   on the same GIST1M R32 / R48 indexes
memory_gist.csv   method,param,qps,recall          HNSW, IVF-Flat, CSPG-R32, CSPG-R48 (in memory)
ablation.csv      dataset,variant,L,qps,mean_ios,recall   DiskANN ("stock"), the paper's MCGI ("full"), and every
                                                          ablation variant of m1_tune/ablate.py
latency.csv       dataset,method,L,mean_ms,p999_ms,recall  GIST1M R32, SIFT1B, Deep1B
"""
import csv
import os
import re

HOME = os.path.expanduser("~")
H = f"{HOME}/data/mcgi"
OUT = f"{HOME}/hpdic/adadisk/paper/NeurIPS26/script/data"


def rows(path, ncol):
    """Numeric result rows of a DiskANN (9 columns) or PipeANN (8 columns) search log."""
    out = []
    for line in open(path, errors="replace"):
        f = line.split()
        if len(f) == ncol and re.match(r"^\d+$", f[0]):
            out.append([float(x) for x in f])
    return out


def diskann(path):  # L, qps, mean_ios, recall, mean_us, p999_us
    return [(r[0], r[2], r[5], r[8], r[3], r[4]) for r in rows(path, 9)]


def pipeann(path):  # L, qps, mean_ios, recall
    return [(r[0], r[2], r[6], r[7]) for r in rows(path, 8)]


def write(name, header, data):
    with open(f"{OUT}/{name}", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(data)
    print(name, len(data))


def main():
    cmp = []
    for R in (32, 48):
        n = f"diskann_R{R}_L100"
        for tag, method in (("stock", "DiskANN"), ("mcgi", "MCGI")):
            cmp += [(R, method, int(L), q, io, rec) for L, q, io, rec, _, _ in diskann(f"{H}/clean/gist_{n}_{tag}.log")]
        for W in (4, 32):
            cmp += [(R, f"PipeANN-W{W}", int(L), q, io, rec)
                    for L, q, io, rec in pipeann(f"{H}/pipeann/gist_R{R}/pipeann_W{W}.log")]
    write("gist_compare.csv", ["R", "method", "L", "qps", "mean_ios", "recall"], cmp)

    mem = [r for r in csv.reader(open(f"{H}/memory_baselines/memory_gist.csv"))][1:]
    for R in (32, 48):
        for r in list(csv.DictReader(open(f"{H}/cspg/cspg_R{R}.csv"))):
            mem.append([f"cspg-R{R}", r["efq"], f"{float(r['qps']):.1f}", f"{100 * float(r['recall']):.2f}"])
    write("memory_gist.csv", ["method", "param", "qps", "recall"], mem)

    abl = []
    for ds, n in (("sift", "diskann_R32_L50"), ("glove", "diskann_R64_L100"), ("gist", "diskann_R32_L100")):
        for tag, v in (("stock", "stock"), ("mcgi", "full")):
            abl += [(ds, v, int(L), q, io, rec) for L, q, io, rec, _, _ in diskann(f"{H}/clean/{ds}_{n}_{tag}.log")]
        for f in sorted(os.listdir(f"{H}/ablation/{ds}")):
            if f.endswith(".log"):
                abl += [(ds, f[:-4], int(L), q, io, rec) for L, q, io, rec, _, _ in diskann(f"{H}/ablation/{ds}/{f}")]
    write("ablation.csv", ["dataset", "variant", "L", "qps", "mean_ios", "recall"], abl)

    lat = []
    src = {"gist": [f"{H}/clean/gist_diskann_R32_L100_{{}}.log"],
           "sift1b": [f"{H}/sift1b/eval/{{}}.log", f"{H}/sift1b/eval/{{}}_hi.log"],
           "deep1b": [f"{H}/deep1b/eval/{{}}.log", f"{H}/deep1b/eval/{{}}_ext.log"]}
    for ds, paths in src.items():
        for tag, method in (("stock", "DiskANN"), ("mcgi", "MCGI")):
            for p in paths:
                lat += [(ds, method, int(L), round(m / 1e3, 3), round(t / 1e3, 3), rec)
                        for L, q, io, rec, m, t in diskann(p.format(tag))]
    write("latency.csv", ["dataset", "method", "L", "mean_ms", "p999_ms", "recall"], lat)


if __name__ == "__main__":
    main()
