import os
import time
import numpy as np
import faiss

# ================= 路径配置 =================
DATA_DIR = os.path.expanduser("~/hpdic/gist_data")

# 修正文件名：对应你 ls 出来的真实文件
BASE_FILE = os.path.join(DATA_DIR, "gist_base.fvecs")
QUERY_FILE = os.path.join(DATA_DIR, "gist_query.fvecs")
GT_FILE = os.path.join(DATA_DIR, "gist_groundtruth.ivecs")

# HNSW 参数
M = 32
efConstruction = 200 
efSearch_list = [32, 64, 100, 200, 400]

def ivecs_read(fname):
    """ 
    读取 .ivecs 格式 (Texmex 标准格式) 
    结构: [dim, v1, v2, ..., dim, v1, ...] (每个向量前都有维度)
    """
    print(f"📖 Reading {fname}...")
    # 1. 作为一个巨大的 int32 数组读入
    a = np.fromfile(fname, dtype='int32')
    
    # 2. 获取维度 d (第一个元素)
    d = a[0]
    
    # 3. Reshape: 每一行是 (d + 1) 个 int32，其中第 1 个是 header
    # 4. Slice: 去掉每行的第 1 个元素 (header)，只留数据
    return a.reshape(-1, d + 1)[:, 1:].copy()

def fvecs_read(fname):
    """ 
    读取 .fvecs 格式 
    逻辑和 ivecs 一样，只是最后转成 float32 视图
    """
    return ivecs_read(fname).view('float32')

def main():
    print(f"📂 数据目录: {DATA_DIR}")
    
    # 1. 加载数据
    if not os.path.exists(BASE_FILE):
        print(f"❌ 找不到文件: {BASE_FILE}")
        return

    # GIST1M Base
    xb = fvecs_read(BASE_FILE)
    N, D = xb.shape
    print(f"✅ Base Loaded: N={N}, D={D} (Expected 960)")

    # Query
    xq = fvecs_read(QUERY_FILE)
    print(f"✅ Query Loaded: N={xq.shape[0]}")

    # Ground Truth
    gt = ivecs_read(GT_FILE)
    print(f"✅ GT Loaded: N={gt.shape[0]}, K={gt.shape[1]}")

    # 2. 建索引
    print(f"\n🏗️  Building HNSW Index (M={M}, ef={efConstruction})...")
    # GIST 是 960维，HNSWFlat 完全没问题
    index = faiss.IndexHNSWFlat(D, M, faiss.METRIC_L2)
    index.hnsw.efConstruction = efConstruction
    
    t0 = time.time()
    index.add(xb)
    print(f"✅ Build done in {time.time()-t0:.2f}s")

    # 3. 搜索
    print(f"\n🔍 Searching...")
    print(f"{'efSearch':<10} | {'Recall@10':<10} | {'QPS':<10} | {'Latency(ms)':<10}")
    print("-" * 50)
    
    for ef in efSearch_list:
        index.hnsw.efSearch = ef
        
        t_start = time.time()
        D_res, I_res = index.search(xq, 10) # Top-10
        duration = time.time() - t_start
        
        qps = xq.shape[0] / duration
        latency = (duration / xq.shape[0]) * 1000
        
        # 算 Recall
        recall_cnt = 0
        # GIST1M 的 GT 通常包含 100 个近邻。
        # 我们这里计算 Recall@10 (Pred) against GT@100 (True) 中的 Top10
        # 为了严格对比，我们通常看 Intersection(Pred@10, GT@10)
        for i in range(xq.shape[0]):
            gt_set = set(gt[i, :10]) # 取 GT 的前 10
            res_set = set(I_res[i])
            recall_cnt += len(gt_set.intersection(res_set))
            
        recall = (recall_cnt / (xq.shape[0] * 10)) * 100
        
        print(f"{ef:<10} | {recall:<10.2f} | {qps:<10.0f} | {latency:<10.3f}")

if __name__ == "__main__":
    main()