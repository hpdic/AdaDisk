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

# 这是一个极其稳定的下载源，来自 ann-benchmarks 官方
URL = "http://ann-benchmarks.com/glove-100-angular.hdf5"
HDF5_FILE = os.path.join(DATA_ROOT, "glove.hdf5")

# 输出文件
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
    # 1. 下载
    download_file(URL, HDF5_FILE)
    
    # 2. 读取 HDF5 并转换
    print("⚙️ Processing HDF5...")
    f = h5py.File(HDF5_FILE, 'r')
    
    # 提取 Base (Train)
    if not os.path.exists(DST_BASE):
        base_data = f['train'][:]
        save_bin(base_data, DST_BASE, 'float')
    
    # 提取 Query (Test)
    if not os.path.exists(DST_QUERY):
        query_data = f['test'][:]
        save_bin(query_data, DST_QUERY, 'float')
        
    # 提取 Ground Truth (Neighbors)
    # 注意：GloVe 是 Angular 距离，但对于归一化向量，L2 排序是一样的。
    # ann-benchmarks 里的 GT 格式直接就是最近邻的 ID
    if not os.path.exists(DST_GT):
        gt_data = f['neighbors'][:]
        # 只需要前 100 个或者前 10 个，通常全部保留
        save_bin(gt_data, DST_GT, 'int')
        
    f.close()
    print("✅ GloVe-100 Ready!")

if __name__ == "__main__":
    process()