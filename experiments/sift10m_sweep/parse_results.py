"""Summarize sweep search logs: QPS and mean IOs interpolated at target recalls.

Usage: python3 parse_results.py [log_dir] > results.csv
"""
import glob
import os
import re
import sys

import numpy as np

LOGS = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/data/mcgi/sift10m/logs")
TARGETS = [90.0, 95.0, 98.0]
ROW = re.compile(r"^\s*(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$")


def read_rows(path):
    rows = []
    with open(path, errors="replace") as f:
        for line in f:
            m = ROW.match(line)
            if m:
                L, W, qps, lat, p999, ios, io_us, cpu, rec = map(float, m.groups())
                rows.append((L, qps, ios, rec))
    return sorted(rows, key=lambda r: r[3])


def interp(rows, target, col):
    rec = np.array([r[3] for r in rows])
    val = np.array([r[col] for r in rows])
    if target < rec[0] or target > rec[-1]:
        return float("nan")
    return float(np.interp(target, rec, val))


def main():
    out = {}
    for path in sorted(glob.glob(f"{LOGS}/*_search.log")):
        name = os.path.basename(path)[: -len("_search.log")]
        rows = read_rows(path)
        if rows:
            out[name] = rows
    base = out.get("base12")
    head = ["config"]
    for t in TARGETS:
        head += [f"qps@{t:g}", f"ios@{t:g}", f"qps_gain@{t:g}"]
    print(",".join(head))
    for name, rows in out.items():
        cells = [name]
        for t in TARGETS:
            q = interp(rows, t, 1)
            io = interp(rows, t, 2)
            gain = q / interp(base, t, 1) if base else float("nan")
            cells += [f"{q:.1f}", f"{io:.1f}", f"{gain:.3f}"]
        print(",".join(cells))


if __name__ == "__main__":
    main()
