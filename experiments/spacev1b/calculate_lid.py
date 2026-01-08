import numpy as np
import faiss
import struct

# === 修改区域 ===
DATA_PATH = "/home/cc/hpdic/spacev1b_data/spacev1b_base.bin"
SAMPLE_SIZE = 10000    # 已改为 1万
DIM = 100              # SpaceV 维度
K_NEIGHBORS = 100      # 保持 100 不变，估算更稳

def load_data(filename, n, d):
    print(f"正在读取 {filename} 的前 {n} 个向量...")
    with open(filename, "rb") as f:
        header = f.read(8)
        # 读取数据
        raw_bytes = f.read(n * d)
        data_int8 = np.frombuffer(raw_bytes, dtype=np.int8)
        data = data_int8.reshape(n, d)
        # 转为 float32 用于 faiss 计算
        return data.astype(np.float32)

def compute_lid_mle(distances, k):
    # 开根号转为 L2 距离
    r = np.sqrt(distances)
    r = np.maximum(r, 1e-10)
    
    # r[:, k] 是第 k 个邻居的距离 (参考半径)
    r_k = r[:, k:k+1]
    # r[:, 1:k] 是第 1 到 k-1 个邻居 (排除自己)
    r_neighbors = r[:, 1:k]
    
    # MLE 公式
    ratio_sum = np.sum(np.log(r_k / r_neighbors), axis=1)
    lid_estimates = (k - 2) / ratio_sum
    return lid_estimates

def main():
    # 1. 加载数据
    data = load_data(DATA_PATH, SAMPLE_SIZE, DIM)
    
    # 2. 构建索引 (1万数据量非常小，瞬间完成)
    index = faiss.IndexFlatL2(DIM)
    index.add(data)
    
    # 3. 搜索
    k_search = K_NEIGHBORS + 1
    distances, _ = index.search(data, k_search)
    
    # 4. 计算
    lids = compute_lid_mle(distances, K_NEIGHBORS)
    
    print("\n" + "="*40)
    print(f"SpaceV-1B (Sample 10k) LID Analysis")
    print("="*40)
    print(f"LID Mean      : {np.mean(lids):.4f}")
    print(f"LID Std       : {np.std(lids):.4f}")
    print(f"LID Min / Max : {np.min(lids):.2f} / {np.max(lids):.2f}")
    print("="*40)

if __name__ == "__main__":
    main()

#
# Output:
#
# (venv) cc@uc-nvme:~/hpdic/AdaDisk/experiments/spacev1b$ python calculate_lid.py 
# 正在读取 /home/cc/hpdic/spacev1b_data/spacev1b_base.bin 的前 10000 个向量...

# ========================================
# SpaceV-1B (Sample 10k) LID Analysis
# ========================================
# LID Mean      : 23.1403
# LID Std       : 6.9409
# LID Min / Max : 3.65 / 53.94
# ========================================       