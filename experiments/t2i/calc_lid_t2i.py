import numpy as np
import faiss
import struct
import os

DATA_PATH = os.path.expanduser("~/hpdic/t2i_data/t2i_base_1M.fbin") 
SAMPLE_SIZE = 10000 
DIM = 200  # T2I 是 200 维
K_NEIGHBORS = 100

def compute_lid_mle(distances, k):
    r = np.sqrt(distances)
    r = np.maximum(r, 1e-10)
    r_k = r[:, k:k+1]
    r_neighbors = r[:, 1:k]
    return (k - 2) / np.sum(np.log(r_k / r_neighbors), axis=1)

def main():
    print(f"Reading {DATA_PATH}...")
    with open(DATA_PATH, "rb") as f:
        header = f.read(8)
        num, d = struct.unpack('ii', header)
        print(f"Header: {num} points, {d} dims")
        
        # 验证维度
        if d != DIM:
            print(f"警告: 维度不对! 期望 {DIM}, 实际 {d}")
        
        data = np.fromfile(f, dtype=np.float32, count=SAMPLE_SIZE * d)
        data = data.reshape(SAMPLE_SIZE, d)

    print("Building Index & Searching...")
    index = faiss.IndexFlatL2(d)
    index.add(data)
    
    D, _ = index.search(data, K_NEIGHBORS + 1)
    
    lids = compute_lid_mle(D, K_NEIGHBORS)
    
    print("\n" + "="*40)
    print(f"T2I-1M LID Analysis")
    print("="*40)
    print(f"LID Mean      : {np.mean(lids):.4f}")
    print(f"LID Std       : {np.std(lids):.4f}")
    print(f"LID Min / Max : {np.min(lids):.2f} / {np.max(lids):.2f}")
    print("="*40)

if __name__ == "__main__":
    main()

# ========================================
# T2I-1M LID Analysis
# ========================================
# LID Mean      : 18.3252
# LID Std       : 6.9964
# LID Min / Max : 4.04 / 56.09
# ========================================