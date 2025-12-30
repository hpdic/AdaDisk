import numpy as np
import faiss
import time
import os
import argparse
import glob

# === 配置基础路径 ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(BASE_DIR, 'data')

def load_bin(filename, dtype=np.float32):
    """ 读取 .bin 格式 [N(int32), D(int32), data...] """
    print(f"Loading BIN file: {filename}...")
    with open(filename, 'rb') as f:
        n_points = np.fromfile(f, dtype=np.int32, count=1)[0]
        dim = np.fromfile(f, dtype=np.int32, count=1)[0]
        data = np.fromfile(f, dtype=dtype, count=n_points * dim)
        return data.reshape(n_points, dim)

def load_fvecs(filename):
    """ 读取 .fvecs 格式 """
    print(f"Loading FVECS file: {filename}...")
    with open(filename, 'rb') as f:
        dim = np.fromfile(f, dtype=np.int32, count=1)[0]
        f.seek(0)
        x = np.fromfile(f, dtype=np.float32).reshape(-1, dim+1)
        return np.ascontiguousarray(x[:, 1:], dtype=np.float32)

def load_ivecs(filename):
    """ 读取 .ivecs 格式 """
    print(f"Loading IVECS file: {filename}...")
    with open(filename, 'rb') as f:
        dim = np.fromfile(f, dtype=np.int32, count=1)[0]
        f.seek(0)
        x = np.fromfile(f, dtype=np.int32).reshape(-1, dim+1)
        return np.ascontiguousarray(x[:, 1:], dtype=np.int32)

def load_data_auto(filename, is_gt=False):
    """ 根据后缀自动选择加载器 """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Critical Error: File not found at {filename}")
        
    if filename.endswith('.bin'):
        # GT 在 bin 里通常也是 int32
        dtype = np.int32 if is_gt else np.float32
        return load_bin(filename, dtype=dtype)
    elif filename.endswith('.fvecs'):
        return load_fvecs(filename)
    elif filename.endswith('.ivecs'):
        return load_ivecs(filename)
    else:
        # 未知后缀默认尝试 fvecs
        print(f"[Warning] Unknown extension for {filename}, trying fvecs mode...")
        return load_fvecs(filename)

def find_file_smart(dataset_dir, keywords):
    """
    智能搜索函数：在 dataset_dir 下查找包含 keywords 中任意关键字的文件
    """
    files = os.listdir(dataset_dir)
    # 优先找 .fvecs, .ivecs, .bin
    candidates = []
    for f in files:
        # 必须匹配所有关键字 (比如 'glove' 和 'base')
        if all(k in f for k in keywords):
            candidates.append(f)
    
    if not candidates:
        raise FileNotFoundError(f"Could not find any file in {dataset_dir} containing {keywords}")
    
    # 如果有多个，优先选短的或者常见的后缀，这里简单取第一个
    # 排序是为了保证确定性
    candidates.sort()
    found_path = os.path.join(dataset_dir, candidates[0])
    print(f"[Smart Search] Auto-resolved {'+'.join(keywords)} -> {candidates[0]}")
    return found_path

def get_dataset_files(dataset_name):
    dataset_dir = os.path.join(DATA_ROOT, dataset_name)
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Directory not found: {dataset_dir}")

    # === 关键修改：智能模糊搜索 ===
    # 不再硬编码 'glove-100-angular' 这种前缀
    # 只要文件名里包含 'base', 'query', 'groundtruth'/'gt' 就能找到
    
    # 1. 找 Base 文件 (包含 'base')
    base_file = find_file_smart(dataset_dir, ['base'])
    
    # 2. 找 Query 文件 (包含 'query')
    query_file = find_file_smart(dataset_dir, ['query'])
    
    # 3. 找 GT 文件 (包含 'groundtruth' 或者 'gt')
    # 尝试两次，先找 groundtruth，找不到再找 gt
    try:
        gt_file = find_file_smart(dataset_dir, ['groundtruth'])
    except FileNotFoundError:
        try:
            gt_file = find_file_smart(dataset_dir, ['gt'])
        except FileNotFoundError:
             # 如果还没找到，可能是 sift_gt.bin 这种带前缀的，再试一次 dataset_name + gt
             gt_file = find_file_smart(dataset_dir, [dataset_name, 'gt'])

    disk_index_file = os.path.join(dataset_dir, f'{dataset_name}_ivf_on_disk.index')
    
    return base_file, query_file, gt_file, disk_index_file

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['sift', 'gist', 'glove'])
    parser.add_argument('--nlist', type=int, default=4096)
    parser.add_argument('--normalize', action='store_true')
    args = parser.parse_args()
    
    dataset = args.dataset
    print(f"=== Running Baseline for Dataset: {dataset.upper()} ===")
    
    # 这一步现在会自动搜索文件，不会报错了
    base_file, query_file, gt_file, disk_index_file = get_dataset_files(dataset)

    # 1. 加载数据
    xb = load_data_auto(base_file, is_gt=False)
    xq = load_data_auto(query_file, is_gt=False)
    gt = load_data_auto(gt_file, is_gt=True)
    
    d = xb.shape[1]
    print(f"Data Loaded: Base={xb.shape}, Query={xq.shape}, GT={gt.shape}")

    if args.normalize or dataset == 'glove':
        print("Normalizing vectors (L2) for Cosine Similarity...")
        faiss.normalize_L2(xb)
        faiss.normalize_L2(xq)

    # 2. 构建索引
    if os.path.exists(disk_index_file):
        print(f"\n[Info] Index file {disk_index_file} exists. Skipping build.")
    else:
        print(f"\n--- Building Index in RAM first ---")
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, args.nlist, faiss.METRIC_L2)
        
        # 训练 (GloVe 数据量大，依然采样 16w)
        train_size = min(160000, xb.shape[0])
        print(f"Training quantizer with {train_size} vectors...")
        start_train = time.time()
        index.train(xb[:train_size])
        print(f"Training done in {time.time() - start_train:.2f}s")

        print("Adding vectors to index...")
        start_add = time.time()
        index.add(xb)
        print(f"Adding done in {time.time() - start_add:.2f}s")

        print(f"Saving index to {disk_index_file}...")
        faiss.write_index(index, disk_index_file)
        del index 

    # 3. 磁盘模式 Benchmark
    print(f"\n--- Loading Index in Disk Mode (mmap) ---")
    index = faiss.read_index(disk_index_file, faiss.IO_FLAG_MMAP)
    print(f"Index loaded. Total vectors: {index.ntotal}")

    print(f"\n=== Benchmark Result: {dataset.upper()} ===")
    print(f"{'nprobe':<10} | {'Recall@1':<10} | {'Recall@10':<10} | {'QPS':<10} | {'Latency(ms)':<15}")
    print("-" * 75)

    probe_list = [1, 5, 10, 20, 40, 50, 80, 100, 200]
    if dataset == 'gist': probe_list.extend([300, 400])

    for nprobe in probe_list:
        index.nprobe = nprobe
        start = time.time()
        D, I = index.search(xq, 100)
        end = time.time()
        
        qps = xq.shape[0] / (end - start)
        latency = (end - start) / xq.shape[0] * 1000
        
        gt_nn = gt[:, 0]
        match_mask = (I[:, :10] == gt_nn[:, None])
        recall_1 = np.sum(gt_nn == I[:, 0]) / xq.shape[0]
        recall_10 = np.any(match_mask, axis=1).sum() / xq.shape[0]
        
        print(f"{nprobe:<10} | {recall_1:.4f}     | {recall_10:.4f}     | {qps:<10.2f} | {latency:<15.4f}")
    
    print("-" * 75)

if __name__ == "__main__":
    main()