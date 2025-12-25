import numpy as np
import sys
import os

## TODO: extend/modify this script to guide the hyperparameter selection, e.g., alpha min/max, R, L, etc.

def get_stats(file_path):
    if not os.path.exists(file_path):
        return None
    data = np.fromfile(file_path, dtype=np.float32)
    # 如果你的二进制文件包含 header，请根据实际情况 data = data[2:]
    return {
        "Mean": np.mean(data),
        "Std": np.std(data),
        "Min": np.min(data),
        "Max": np.max(data),
        "Median": np.median(data),
        "Count": len(data)
    }

def print_table(files):
    header = f"{'Dataset':<15} | {'Mean':<8} | {'Std':<8} | {'Min':<8} | {'Max':<8} | {'Median':<8}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    
    for name, path in files.items():
        stats = get_stats(path)
        if stats:
            print(f"{name:<15} | {stats['Mean']:<8.2f} | {stats['Std']:<8.2f} | {stats['Min']:<8.2f} | {stats['Max']:<8.2f} | {stats['Median']:<8.2f}")
        else:
            print(f"{name:<15} | File not found")
    print("-" * len(header))

if __name__ == "__main__":
    base_path = "/home/cc/AdaDisk/experiments/data"
    files_to_analyze = {
        "SIFT1M": f"{base_path}/sift/sift_lid.bin",
        "GloVe1M": f"{base_path}/glove/glove_lid.bin",
        "GIST1M": f"{base_path}/gist/gist_lid.bin"
    }
    print_table(files_to_analyze)