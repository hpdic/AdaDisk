import numpy as np
import faiss
import time
import os
import argparse

# === 配置基础路径 ===
# 假设脚本在 experiments/scripts/，数据在 experiments/data/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(BASE_DIR, 'data')

def get_dataset_files(dataset_name):
    """
    根据数据集名字生成文件路径
    假设目录结构: data/{dataset}/{dataset}_base.fvecs
    """
    dataset_dir = os.path.join(DATA_ROOT, dataset_name)
    
    # 这里定义不同数据集的文件名前缀规则
    # 如果你的文件名比较特殊，可以在这里修改
    prefix_map = {
        'sift': 'sift',
        'gist': 'gist',
        'glove': 'glove' # 注意检查 glove 文件夹下的具体文件名
    }
    
    prefix = prefix_map.get(dataset_name, dataset_name)
    
    base_file = os.path.join(dataset_dir, f'{prefix}_base.fvecs')
    query_file = os.path.join(dataset_dir, f'{prefix}_query.fvecs')
    gt_file = os.path.join(dataset_dir, f'{prefix}_groundtruth.ivecs')
    disk_index_file = os.path.join(dataset_dir, f'{dataset_name}_ivf_on_disk.index')
    
    return base_file, query_file, gt_file, disk_index_file

def load_fvecs(filename):
    print(f"Loading {filename}...")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
        
    with open(filename, 'rb') as f:
        dim = np.fromfile(f, dtype=np.int32, count=1)[0]
        f.seek(0)
        x = np.fromfile(f, dtype=np.float32)
        x = x.reshape(-1, dim+1)
        return np.ascontiguousarray(x[:, 1:], dtype=np.float32)

def load_ivecs(filename):
    print(f"Loading {filename}...")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    with open(filename, 'rb') as f:
        dim = np.fromfile(f, dtype=np.int32, count=1)[0]
        f.seek(0)
        x = np.fromfile(f, dtype=np.int32)
        x = x.reshape(-1, dim+1)
        return np.ascontiguousarray(x[:, 1:], dtype=np.int32)

def main():
    parser = argparse.ArgumentParser(description="Run Faiss On-Disk Baseline")
    parser.add_argument('--dataset', type=str, required=True, choices=['sift', 'gist', 'glove'], help='Dataset name (e.g., sift, gist, glove)')
    parser.add_argument('--nlist', type=int, default=4096, help='Number of inverted lists (clusters)')
    parser.add_argument('--normalize', action='store_true', help='Normalize vectors (useful for GloVe/Cosine similarity)')
    args = parser.parse_args()

    dataset = args.dataset
    print(f"=== Running Baseline for Dataset: {dataset.upper()} ===")

    # 1. 获取路径
    base_file, query_file, gt_file, disk_index_file = get_dataset_files(dataset)

    # 2. 加载数据
    try:
        xb = load_fvecs(base_file)
        xq = load_fvecs(query_file)
        gt = load_ivecs(gt_file)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Please check if files exist in {os.path.join(DATA_ROOT, dataset)}")
        return

    # GloVe 特殊处理：如果需要 Cosine 距离，通常做法是归一化后用 L2
    if args.normalize or dataset == 'glove':
        print("Normalizing vectors for Cosine Similarity equivalence...")
        faiss.normalize_L2(xb)
        faiss.normalize_L2(xq)

    d = xb.shape[1]
    print(f"Base shape: {xb.shape}, Dim: {d}")
    print(f"Query shape: {xq.shape}")

    # 3. 训练量化器
    print(f"\n--- Training Quantizer (nlist={args.nlist}) ---")
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, args.nlist, faiss.METRIC_L2)
    
    start_train = time.time()
    # 为了加速训练，如果数据量太大 (比如 > 100万)，可以只采样一部分来训练
    train_size = min(1000000, xb.shape[0])
    index.train(xb[:train_size]) 
    print(f"Training time: {time.time() - start_train:.2f}s")

    # 4. 创建 On-Disk 索引结构
    print(f"\n--- Creating On-Disk Index ---")
    faiss.write_index(index, disk_index_file)

    # 5. 重新加载为 MMAP 模式
    print("Reloading index with IO_FLAG_MMAP (Disk Mode)...")
    index = faiss.read_index(disk_index_file, faiss.IO_FLAG_MMAP)

    # 6. 添加数据 (Batch Add)
    print("Adding vectors to disk index...")
    batch_size = 100000
    for i in range(0, xb.shape[0], batch_size):
        end = min(i + batch_size, xb.shape[0])
        index.add(xb[i:end])
        print(f"  Added {end}/{xb.shape[0]}...", end='\r')
    print(f"\nIndex built. Total: {index.ntotal}")

    # 7. Benchmark
    print(f"\n=== Benchmark Result: {dataset.upper()} ===")
    print(f"{'nprobe':<10} | {'Recall@1':<10} | {'Recall@10':<10} | {'QPS':<10} | {'Latency(ms)':<15}")
    print("-" * 75)

    probe_list = [1, 5, 10, 20, 40, 50, 80, 100, 200]
    
    # 针对 GIST 这种高维数据，nprobe 可能需要更大才能达到高 Recall
    if dataset == 'gist':
        probe_list.extend([300, 400])

    for nprobe in probe_list:
        index.nprobe = nprobe
        
        start_time = time.time()
        D, I = index.search(xq, 100) # Search Top-100
        end_time = time.time()
        
        total_time = end_time - start_time
        qps = xq.shape[0] / total_time
        latency = (total_time / xq.shape[0]) * 1000
        
        # Calculate Recall
        # 兼容不同形状的 GT (有些是 (N, 100), 有些是 (N, 1))
        # 我们只看 GT 的第一列（Nearest Neighbor）
        gt_nn = gt[:, 0]
        
        # Recall@1
        correct_1 = np.sum(gt_nn == I[:, 0])
        recall_1 = correct_1 / xq.shape[0]
        
        # Recall@10
        # 使用 numpy 广播机制加速计算
        # 检查 gt_nn 是否在 I[:, :10] 的每一行中
        # 这种写法比 for loop 快很多
        match_mask = (I[:, :10] == gt_nn[:, None])
        correct_10 = np.any(match_mask, axis=1).sum()
        recall_10 = correct_10 / xq.shape[0]
        
        print(f"{nprobe:<10} | {recall_1:.4f}     | {recall_10:.4f}     | {qps:<10.2f} | {latency:<15.4f}")

    print("-" * 75)

if __name__ == "__main__":
    main()