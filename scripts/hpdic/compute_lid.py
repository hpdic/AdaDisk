"""
@file: compute_lid.py
@brief: Calculates REAL LID using MLE estimation based on k-NN distances.
        Replaces the synthetic random data.
"""

import numpy as np
import struct
import os
import time
from sklearn.neighbors import NearestNeighbors

# 配置路径
DATA_DIR = "hpdic_data"
RAW_FILE = os.path.join(DATA_DIR, "ingest_raw.bin")
LID_FILE = os.path.join(DATA_DIR, "ingest_lid.bin")

# 算法参数
K_NEIGHBORS = 20  # 计算 LID 通常取 k=10~30 之间

def read_diskann_bin(filepath):
    """读取 DiskANN float bin 文件"""
    print(f"Reading raw data from {filepath}...")
    file_size = os.path.getsize(filepath)
    with open(filepath, "rb") as f:
        npts = struct.unpack("i", f.read(4))[0]
        dim = struct.unpack("i", f.read(4))[0]
        
        # 剩下的全是 float32
        data = np.frombuffer(f.read(), dtype=np.float32)
        data = data.reshape(npts, dim)
    
    print(f"Loaded shape: {data.shape}")
    return data, npts, dim

def save_diskann_bin(filename, data):
    """保存为 DiskANN bin 格式"""
    npts = len(data)
    dim = 1
    print(f"Saving LID to {filename}...")
    with open(filename, "wb") as f:
        f.write(struct.pack("i", npts))
        f.write(struct.pack("i", dim))
        f.write(data.astype(np.float32).tobytes())
    print("Done.")

def compute_lid_mle(data, k=20):
    """
    使用 Levina-Bickel MLE 估算器计算 LID。
    LID(x) = (k-1) / sum( ln(r_k / r_j) ) for j=1..k-1
    """
    print(f"Computing k-NN (k={k}) for LID estimation...")
    t0 = time.time()
    
    # 1. 寻找 k 个最近邻 (需要 k+1，因为第1个是自己，距离为0)
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto', n_jobs=-1).fit(data)
    distances, _ = nbrs.kneighbors(data)
    
    # distances[:, 0] 是自己到自己的距离 (0.0)，去掉它
    # 取第 1 到 k 个邻居的距离
    knn_dists = distances[:, 1:] 
    
    # 防止距离为0 (重复点) 导致除零错误，加一个极小值
    knn_dists = np.maximum(knn_dists, 1e-10)

    # 2. 应用 MLE 公式
    # r_k 是第 k 个邻居的距离 (也就是 knn_dists 的最后一列)
    r_k = knn_dists[:, -1]
    
    # 计算 sum( ln(r_k / r_j) )
    # 利用 log(a/b) = log(a) - log(b)
    # 也就是 k * ln(r_k) - sum(ln(r_j))
    
    log_sum = np.sum(np.log(r_k[:, None] / knn_dists[:, :-1]), axis=1)
    
    # MLE 估计值
    lid_estimates = (k - 1) / log_sum
    
    t1 = time.time()
    print(f"LID calculated in {t1-t0:.2f} seconds.")
    
    # 3. 简单的后处理：限制极其异常的值 (比如 LID > 100 对于 128维数据可能不太正常)
    lid_estimates = np.clip(lid_estimates, 0.1, 200.0)
    
    return lid_estimates

def main():
    if not os.path.exists(RAW_FILE):
        print(f"Error: Raw file {RAW_FILE} not found.")
        return

    # 1. 读取数据
    data, npts, dim = read_diskann_bin(RAW_FILE)

    # 2. 计算真实 LID
    real_lid = compute_lid_mle(data, k=K_NEIGHBORS)

    # 3. 打印统计信息 (看看真实数据的分布)
    print("-" * 40)
    print(f"Real LID Stats (k={K_NEIGHBORS}):")
    print(f"  Mean:   {np.mean(real_lid):.4f}")
    print(f"  StdDev: {np.std(real_lid):.4f}")
    print(f"  Min:    {np.min(real_lid):.4f}")
    print(f"  Max:    {np.max(real_lid):.4f}")
    print("-" * 40)

    # 4. 保存
    save_diskann_bin(LID_FILE, real_lid)

if __name__ == "__main__":
    main()
 