import os
import subprocess
import csv
import re
import shutil
import time

# ================= 配置区域 =================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BUILDER_BIN = os.path.join(PROJECT_ROOT, "build", "apps", "build_disk_index")
SEARCHER_BIN = os.path.join(PROJECT_ROOT, "build", "apps", "search_disk_index")

# [修正] 文件名改回 smart_scan_patch.csv，不再随便加后缀
NEW_CSV_PATH = os.path.join(PROJECT_ROOT, "experiments", "smart_scan_patch.csv")

# 临时目录
TEMP_DIR = os.path.join(PROJECT_ROOT, "experiments", "temp_smart_indices_v3")

COMMON_ALPHA_CONFIGS = [
    # --- Min = 1.0 的组 (之前漏了 1.3 到 2.0) ---
    (1.0, 1.1), 
    (1.0, 1.2), 
    (1.0, 1.3), 
    (1.0, 1.4), 
    (1.0, 1.5), 
    (1.0, 1.6), 
    (1.0, 1.7), 
    (1.0, 2.0),

    # --- Min = 1.1 的组 ---
    (1.1, 1.2), 
    (1.1, 1.3), 
    (1.1, 1.4), 
    (1.1, 1.5), 
    (1.1, 1.6), 
    (1.1, 1.7), 
    (1.1, 2.0)
]

TASKS = [
    # --- Task 1: SIFT ---
    {
        "dataset": "sift",
        "R_list": [32, 48],
        "L_search_list": [10, 20, 30, 40], 
        "alpha_configs": COMMON_ALPHA_CONFIGS,
        "run_baseline": True 
    },
    # --- Task 2: GloVe ---
    {
        "dataset": "glove",
        "R_list": [80], 
        "L_search_list": [50, 100, 150, 200],
        "alpha_configs": COMMON_ALPHA_CONFIGS,
        "run_baseline": True
    },
    # --- Task 3: GIST ---
    {
        "dataset": "gist",
        "R_list": [48, 64],
        "L_search_list": [75],
        "alpha_configs": COMMON_ALPHA_CONFIGS,
        "run_baseline": True
    }
]

# ===========================================

def run_command(cmd_list, log_file=None):
    try:
        if log_file:
            with open(log_file, "w") as f:
                subprocess.check_call(cmd_list, stdout=f, stderr=subprocess.STDOUT)
        else:
            result = subprocess.run(cmd_list, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"  [Error] Command failed: {e}")
        return None

def parse_and_save(dataset, R, amin, amax, algo, L, output, writer, csv_file):
    if output is None: return
    try:
        lines = output.strip().split('\n')
        for line in reversed(lines):
            parts = line.split()
            if len(parts) >= 4 and re.match(r'^\d+', parts[0]):
                qps = parts[1]
                lat = parts[2]
                recall = parts.get(8, parts[-1]) 
                
                row = [dataset, R, amin, amax, algo, L, qps, lat, recall]
                writer.writerow(row)
                csv_file.flush()
                print(f"    -> [{algo}] L={L}: QPS={qps}, Recall={recall}")
                return
        print(f"    -> [{algo}] L={L}: [Parse Error] No stats found.")
    except Exception as e:
        print(f"    -> [{algo}] L={L}: [Error] {e}")

def run_build_and_search(dataset, R, L_list, is_mcgi, amin, amax, writer, csv_file, data_dir):
    mode_name = "MCGI" if is_mcgi else "Baseline"
    # 临时索引文件名格式: idx_dataset_R_min_max
    alpha_str = f"{amin}_{amax}" if is_mcgi else "base"
    index_prefix = os.path.join(TEMP_DIR, f"idx_{dataset}_{R}_{alpha_str}")
    
    # 1. Build
    build_cmd = [
        BUILDER_BIN,
        "--data_type", "float",
        "--dist_fn", "l2",
        "--data_path", os.path.join(data_dir, f"{dataset}_base.bin"),
        "--index_path_prefix", index_prefix,
        "-R", str(R),
        "-L", "100",
        "-B", "0.1",
        "-M", "1.0",
        "-T", "32"
    ]
    
    if is_mcgi:
        build_cmd.extend([
            "--use_mcgi",
            "--lid_path", os.path.join(data_dir, f"{dataset}_lid.bin"),
            "--alpha_min", str(amin),
            "--alpha_max", str(amax)
        ])
    
    print(f"    Building {mode_name}...", end=" ", flush=True)
    start_t = time.time()
    run_command(build_cmd, log_file=os.path.join(TEMP_DIR, "build.log"))
    
    if not os.path.exists(index_prefix + "_disk.index"):
        print("Failed!")
        return
    print(f"Done ({time.time()-start_t:.1f}s)")

    # 2. Search
    for L in L_list:
        search_cmd = [
            SEARCHER_BIN,
            "--data_type", "float",
            "--dist_fn", "l2",
            "--index_path_prefix", index_prefix,
            "--query_file", os.path.join(data_dir, f"{dataset}_query.bin"),
            "--gt_file", os.path.join(data_dir, f"{dataset}_gt.bin"),
            "-K", "10",
            "-L", str(L),
            "--result_path", os.path.join(TEMP_DIR, "res"),
            "--num_threads", "32"
        ]
        out = run_command(search_cmd)
        
        log_amin = amin if is_mcgi else 1.0 
        log_amax = amax if is_mcgi else 1.0 
        
        parse_and_save(dataset, R, log_amin, log_amax, mode_name, L, out, writer, csv_file)

    # 3. Cleanup (删除 idx_ 开头的临时文件)
    for f in os.listdir(TEMP_DIR):
        if f.startswith(f"idx_{dataset}"):
            try:
                os.remove(os.path.join(TEMP_DIR, f))
            except: pass

def main():
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    file_exists = os.path.exists(NEW_CSV_PATH)
    csv_file = open(NEW_CSV_PATH, 'a', newline='')
    writer = csv.writer(csv_file)
    if not file_exists:
        writer.writerow(['Dataset', 'R', 'Alpha_Min', 'Alpha_Max', 'Algorithm', 'L', 'QPS', 'Latency', 'Recall'])
        csv_file.flush()

    print(f"Starting Smart Scan -> {NEW_CSV_PATH}")

    for task in TASKS:
        dataset = task['dataset']
        data_dir = os.path.join(PROJECT_ROOT, "experiments", "data", dataset)
        
        for R in task['R_list']:
            print(f"\n>>> Task: {dataset} R={R}")

            # 先跑 Baseline
            if task.get('run_baseline', False):
                print(f"  [Baseline Run] {dataset} R={R}")
                run_build_and_search(dataset, R, task['L_search_list'], False, 1.0, 1.0, writer, csv_file, data_dir)

            # 再跑 MCGI
            for (amin, amax) in task['alpha_configs']:
                if amin >= amax: continue
                print(f"  [MCGI Run] Alpha=[{amin}-{amax}]")
                run_build_and_search(dataset, R, task['L_search_list'], True, amin, amax, writer, csv_file, data_dir)

    csv_file.close()
    print("\nAll Done.")

if __name__ == "__main__":
    main()