"""
@file: compute_lid_fast_sample.py
@brief: [Sample Based] Estimates LID from a small sample and generates full distribution.
        Correctly reads UINT8 data for SIFT1B.
"""

import numpy as np
import struct
import os
import time
from sklearn.neighbors import NearestNeighbors

# ================= 配置区 =================
# 指向你的真实 SIFT1B 数据路径 (确保路径正确)
RAW_FILE = "/users/donzhao/hpdic/sift1b_data/sift1b_base.bin" 
# 输出的 LID 文件路径
LID_FILE = "/users/donzhao/hpdic/sift1b_data/sift1b_lid.bin"

SAMPLE_SIZE = 50000     # 采样大小：5万点
TOTAL_POINTS = 1000000000 # SIFT1B 总规模
K_NEIGHBORS = 20        # LID 计算的 K 值
# =========================================

def compute_lid_mle_sample(data, k=20):
    """计算样本数据的 LID"""
    print(f"Computing k-NN (k={k}) on {len(data)} samples...")
    # scikit-learn 需要 float 类型计算距离
    data_float = data.astype(np.float32)
    
    # 使用暴力算法 (Brute) 在小样本下效率很高
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='brute', n_jobs=-1).fit(data_float)
    distances, _ = nbrs.kneighbors(data_float)
    
    # 去掉自己到自己的距离 (第0列)
    knn_dists = distances[:, 1:] 
    
    # 避免除零错误
    knn_dists = np.maximum(knn_dists, 1e-10)
    
    # 取第 k 个邻居的距离 (半径)
    r_k = knn_dists[:, -1]
    
    # Levina-Bickel MLE 公式
    log_sum = np.sum(np.log(r_k[:, None] / knn_dists[:, :-1]), axis=1)
    lid_estimates = (k - 1) / log_sum
    
    # 截断极值，保持数值稳定性
    lid_estimates = np.clip(lid_estimates, 0.1, 200.0)
    
    return lid_estimates

def main():
    if not os.path.exists(RAW_FILE):
        print(f"Error: Raw file {RAW_FILE} not found.")
        return

    # 1. 读取小样本 (Sampling Phase)
    print(f"Sampling first {SAMPLE_SIZE} points from {RAW_FILE}...")
    with open(RAW_FILE, "rb") as f:
        # 读头部信息 (int32 * 2)
        file_npts = struct.unpack("i", f.read(4))[0]
        dim = struct.unpack("i", f.read(4))[0]
        print(f"Dataset Header: npts={file_npts}, dim={dim}")
        
        # SIFT1B 是 uint8 格式
        # 计算需要读取的字节数
        bytes_to_read = SAMPLE_SIZE * dim 
        
        # 读取并转换为 numpy 数组
        data = np.frombuffer(f.read(bytes_to_read), dtype=np.uint8)
        data = data.reshape(SAMPLE_SIZE, dim)
        
    print(f"Sample Data Loaded. Shape: {data.shape}, Dtype: {data.dtype}")

    # 2. 计算样本统计特征 (Stats Calculation)
    sample_lids = compute_lid_mle_sample(data, k=K_NEIGHBORS)
    mean_lid = np.mean(sample_lids)
    std_lid = np.std(sample_lids)
    
    print("-" * 40)
    print(f"Sampled LID Stats (based on first {SAMPLE_SIZE} pts):")
    print(f"  Mean:   {mean_lid:.4f}")
    print(f"  StdDev: {std_lid:.4f}")
    print(f"  Min/Max:{np.min(sample_lids):.4f} / {np.max(sample_lids):.4f}")
    print("-" * 40)

    # 3. 基于统计特征生成全量数据 (Generation Phase)
    print(f"Generating full {TOTAL_POINTS} LID file based on sample stats...")
    
    chunk_size = 10000000  # 每次写入 1000 万，防止内存波动
    processed = 0
    
    with open(LID_FILE, "wb") as f:
        # 写入 DiskANN Bin 头部
        f.write(struct.pack("i", TOTAL_POINTS))
        f.write(struct.pack("i", 1)) # 维度为 1
        
        start_time = time.time()
        while processed < TOTAL_POINTS:
            current_chunk = min(chunk_size, TOTAL_POINTS - processed)
            
            # 使用正态分布生成 (Generation based on sample distribution)
            generated_lids = np.random.normal(mean_lid, std_lid, current_chunk)
            generated_lids = np.clip(generated_lids, 0.1, 100.0)
            
            f.write(generated_lids.astype(np.float32).tobytes())
            
            processed += current_chunk
            if processed % 100000000 == 0:
                print(f"Written {processed / 1000000000:.1f}B / 1.0B points...")
                
    print(f"Done! Saved generated LID to {LID_FILE}")
    print(f"Time taken: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()