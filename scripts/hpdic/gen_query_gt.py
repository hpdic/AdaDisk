"""
@file: gen_query_gt.py
@brief: Generates Query vectors and Ground Truth (indices) for recall calculation.
"""
import numpy as np
import struct
import os
from sklearn.neighbors import NearestNeighbors

# 配置路径 (与之前保持一致)
DATA_DIR = "hpdic_data"
RAW_FILE = os.path.join(DATA_DIR, "ingest_raw.bin")
QUERY_FILE = os.path.join(DATA_DIR, "ingest_query.bin")
GT_FILE = os.path.join(DATA_DIR, "ingest_gt.bin")

# 参数
NUM_QUERIES = 100   # 生成 100 个查询向量
TOP_K = 100         # 每个查询找 Top-100 真值

def read_f32_bin(filepath):
    with open(filepath, "rb") as f:
        npts = struct.unpack("i", f.read(4))[0]
        dim = struct.unpack("i", f.read(4))[0]
        data = np.frombuffer(f.read(), dtype=np.float32).reshape(npts, dim)
    return data, npts, dim

def save_bin(filename, data, type_code):
    """
    type_code: 'f' for float32 (query), 'I' for uint32 (GT)
    DiskANN GT format: [num_queries] [k] [id_0, id_1... id_k]...
    """
    npts, dim = data.shape
    print(f"Saving {filename} ({npts}x{dim})...")
    with open(filename, "wb") as f:
        f.write(struct.pack("i", npts))
        f.write(struct.pack("i", dim))
        if type_code == 'f':
            f.write(data.astype(np.float32).tobytes())
        elif type_code == 'I':
            f.write(data.astype(np.uint32).tobytes())

def main():
    # 1. 读取底库
    if not os.path.exists(RAW_FILE):
        print("Raw data not found.")
        return
    base_data, npts, dim = read_f32_bin(RAW_FILE)
    print(f"Base data: {npts} points, {dim} dim")

    # 2. 生成查询向量 (从数据分布中采样，或者加高斯噪声)
    # 这里简单起见，随机生成与底库相同分布的数据
    print("Generating queries...")
    # 假设底库是正态分布，我们也生成正态分布，或者直接从底库里切一部分出来做 query (更真实)
    # 方法A: 切片 (容易造成距离为0的完美匹配) -> 不推荐
    # 方法B: 随机生成 (模拟全新用户) -> 推荐
    mean = np.mean(base_data, axis=0)
    std = np.std(base_data, axis=0)
    queries = np.random.normal(mean, std, (NUM_QUERIES, dim)).astype(np.float32)
    save_bin(QUERY_FILE, queries, 'f')

    # 3. 计算 Ground Truth (暴力搜索)
    print(f"Calculating Ground Truth for {NUM_QUERIES} queries (Top-{TOP_K})...")
    nbrs = NearestNeighbors(n_neighbors=TOP_K, algorithm='brute', metric='l2').fit(base_data)
    distances, indices = nbrs.kneighbors(queries)
    
    # 保存 GT (DiskANN 只需要 indices，不需要 distances)
    save_bin(GT_FILE, indices, 'I')
    print("Done.")

if __name__ == "__main__":
    main()