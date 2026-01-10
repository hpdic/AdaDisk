import numpy as np
import faiss
import struct
import os

# === 刚下好的 1M 数据 ===
DATA_PATH = "deep1b_base_1M.fbin" 
SAMPLE_SIZE = 10000   # 采样 1万个点
DIM = 96              # Deep1B 是 96 维
K_NEIGHBORS = 100

def compute_lid_mle(distances, k):
    """ MLE LID 估算 """
    r = np.sqrt(distances)
    r = np.maximum(r, 1e-10)
    # r[:, k] 是第 k 个邻居的距离 (参考半径)
    r_k = r[:, k:k+1]
    # r[:, 1:k] 是第 1 到 k-1 个邻居
    r_neighbors = r[:, 1:k]
    
    ratio_sum = np.sum(np.log(r_k / r_neighbors), axis=1)
    return (k - 2) / ratio_sum

def main():
    if not os.path.exists(DATA_PATH):
        print(f"❌ 找不到文件: {DATA_PATH}")
        return

    print(f"正在读取 {DATA_PATH} ...")
    with open(DATA_PATH, "rb") as f:
        # 读取头: num_points(int), dim(int)
        header = f.read(8)
        num, d = struct.unpack('ii', header)
        print(f"Header: {num} vectors, {d} dimensions")
        
        if d != DIM:
            print(f"⚠️ 维度不匹配！预期 {DIM}，实际 {d}")
        
        # 读取前 SAMPLE_SIZE 个向量 (float32)
        # float32 = 4 bytes
        data = np.fromfile(f, dtype=np.float32, count=SAMPLE_SIZE * d)
        data = data.reshape(SAMPLE_SIZE, d)

    print("构建索引 & 搜索 KNN...")
    index = faiss.IndexFlatL2(d)
    index.add(data)
    
    # 搜 K+1 个 (包含自己)
    distances, _ = index.search(data, K_NEIGHBORS + 1)
    
    print("计算 LID...")
    lids = compute_lid_mle(distances, K_NEIGHBORS)
    
    print("\n" + "="*40)
    print(f"Deep1B (1M Subset) LID Analysis")
    print("="*40)
    print(f"LID Mean      : {np.mean(lids):.4f}")
    print(f"LID Std       : {np.std(lids):.4f}")
    print(f"LID Min / Max : {np.min(lids):.2f} / {np.max(lids):.2f}")
    print("="*40)

if __name__ == "__main__":
    main()

# (venv) cc@uc-nvme:~/hpdic/deep1b_data$ python calc_lid_deep1m.py 
# 正在读取 deep1b_base_1M.fbin ...
# Header: 9990000 vectors, 96 dimensions
# 构建索引 & 搜索 KNN...
# 计算 LID...

# ========================================
# Deep1B (1M Subset) LID Analysis
# ========================================
# LID Mean      : 16.5682
# LID Std       : 5.9916
# LID Min / Max : 1.75 / 39.54
# ========================================    