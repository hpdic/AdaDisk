import os
import struct
import numpy as np
import requests
import h5py
from tqdm import tqdm

# ================= 配置 =================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(SCRIPT_DIR, "../data/glove")
os.makedirs(DATA_ROOT, exist_ok=True)

URL = "http://ann-benchmarks.com/glove-100-angular.hdf5"
HDF5_FILE = os.path.join(DATA_ROOT, "glove.hdf5")

DST_BASE = os.path.join(DATA_ROOT, "glove_base.bin")
DST_QUERY = os.path.join(DATA_ROOT, "glove_query.bin")
DST_GT = os.path.join(DATA_ROOT, "glove_gt.bin")

def download_file(url, dest):
    if os.path.exists(dest):
        print(f"✅ {os.path.basename(dest)} already downloaded.")
        return
    print(f"⬇️  Downloading GloVe-100 from {url}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    with open(dest, 'wb') as f, tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))
    print("Download complete.")

def normalize(data):
    """关键修复：将向量归一化，使得 L2 距离等价于 Cosine 距离"""
    print("⚖️ Normalizing vectors...")
    norm = np.linalg.norm(data, axis=1, keepdims=True)
    # 防止除以 0
    norm[norm == 0] = 1.0 
    return data / norm

def save_bin(data, filename, dtype='float'):
    print(f"💾 Converting to DiskANN bin: {filename} {data.shape}...")
    with open(filename, "wb") as f:
        npts, dim = data.shape
        f.write(struct.pack("i", npts))
        f.write(struct.pack("i", dim))
        if dtype == 'float':
            f.write(data.astype(np.float32).tobytes())
        else:
            f.write(data.astype(np.uint32).tobytes())

def process():
    download_file(URL, HDF5_FILE)
    
    print("⚙️ Processing HDF5...")
    f = h5py.File(HDF5_FILE, 'r')
    
    # Base (Train) - 必须归一化
    if not os.path.exists(DST_BASE):
        base_data = f['train'][:]
        base_data = normalize(base_data) # <--- FIX
        save_bin(base_data, DST_BASE, 'float')
    
    # Query (Test) - 必须归一化
    if not os.path.exists(DST_QUERY):
        query_data = f['test'][:]
        query_data = normalize(query_data) # <--- FIX
        save_bin(query_data, DST_QUERY, 'float')
        
    # GT
    if not os.path.exists(DST_GT):
        gt_data = f['neighbors'][:]
        save_bin(gt_data, DST_GT, 'int')
        
    f.close()
    print("✅ GloVe-100 Ready (Normalized)!")

if __name__ == "__main__":
    process()