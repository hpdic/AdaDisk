import os
import struct
import numpy as np
import urllib.request
import tarfile

# 设置数据根目录 (相对于脚本位置)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(SCRIPT_DIR, "../data")

DATASETS = {
    "sift": {
        "url": "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz",
        "dim": 128,
        "base": "sift/sift_base.fvecs",
        "query": "sift/sift_query.fvecs",
        "gt": "sift/sift_groundtruth.ivecs"
    },
    "gist": {
        "url": "ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz",
        "dim": 960,
        "base": "gist/gist_base.fvecs",
        "query": "gist/gist_query.fvecs",
        "gt": "gist/gist_groundtruth.ivecs"
    }
}

def read_fvecs(filename):
    print(f"Reading fvecs: {filename}")
    fv = np.fromfile(filename, dtype=np.float32)
    if fv.size == 0: return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    fv = fv.reshape(-1, 1 + dim)
    return fv[:, 1:]

def read_ivecs(filename):
    print(f"Reading ivecs: {filename}")
    iv = np.fromfile(filename, dtype=np.int32)
    if iv.size == 0: return np.zeros((0, 0))
    dim = iv.view(np.int32)[0]
    iv = iv.reshape(-1, 1 + dim)
    return iv[:, 1:]

def save_bin(data, filename, type_code):
    print(f"Saving bin: {filename}")
    with open(filename, "wb") as f:
        npts, dim = data.shape
        f.write(struct.pack("i", npts))
        f.write(struct.pack("i", dim))
        if type_code == 'float':
            f.write(data.astype(np.float32).tobytes())
        else:
            f.write(data.astype(np.uint32).tobytes())

def download_and_extract(name, url):
    dest_dir = os.path.join(DATA_ROOT, name)
    os.makedirs(dest_dir, exist_ok=True)
    tar_path = os.path.join(DATA_ROOT, os.path.basename(url))
    
    if not os.path.exists(tar_path):
        print(f"Downloading {name}...")
        urllib.request.urlretrieve(url, tar_path)
    
    print(f"Extracting {name}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=DATA_ROOT)

def process(name):
    info = DATASETS[name]
    download_and_extract(name, info["url"])
    
    base_src = os.path.join(DATA_ROOT, info["base"])
    query_src = os.path.join(DATA_ROOT, info["query"])
    gt_src = os.path.join(DATA_ROOT, info["gt"])
    
    out_dir = os.path.join(DATA_ROOT, name) # Clean output folder
    
    # Base
    base_data = read_fvecs(base_src)
    save_bin(base_data, os.path.join(out_dir, f"{name}_base.bin"), 'float')
    
    # Query
    query_data = read_fvecs(query_src)
    save_bin(query_data, os.path.join(out_dir, f"{name}_query.bin"), 'float')
    
    # GT
    gt_data = read_ivecs(gt_src)
    save_bin(gt_data, os.path.join(out_dir, f"{name}_gt.bin"), 'int')
    
    print(f"Dataset {name} prepared in {out_dir}")

if __name__ == "__main__":
    process("sift")
    process("gist") # Uncomment to run GIST