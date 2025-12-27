import os
import re
import csv
import glob

# ================= 配置区 =================
# 输入文件夹 (存放 txt 结果的地方)
INPUT_DIR = "experiments/fullscan"
# 输出文件路径
OUTPUT_CSV = "experiments/grid_search_summary.csv"

def parse_filename(filename):
    """
    从文件名解析参数
    例如: gist_R32_min1.0_max1.1.txt
    """
    basename = os.path.basename(filename)
    # 正则匹配: [Dataset]_R[R]_min[Min]_max[Max].txt
    match = re.search(r'([a-zA-Z0-9]+)_R(\d+)_min([\d\.]+)_max([\d\.]+)\.txt', basename)
    if match:
        return {
            'dataset': match.group(1),
            'R': int(match.group(2)),
            'alpha_min': float(match.group(3)),
            'alpha_max': float(match.group(4))
        }
    return None

def parse_file_content(filepath, metadata):
    """
    读取文件内容，提取表格数据
    """
    results = []
    current_algo = None # Baseline or MCGI
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                
                # 1. 识别当前是哪个算法
                if "--- Baseline ---" in line:
                    current_algo = "Baseline"
                    continue
                elif "--- MCGI ---" in line:
                    current_algo = "MCGI"
                    continue
                
                # 2. 识别数据行 (以数字开头)
                # 格式: L  QPS  Lat(us)  Recall
                # 例如: 50   1234.5   123.4    0.95
                if current_algo and re.match(r'^\d+', line):
                    parts = line.split()
                    if len(parts) >= 4:
                        # 有时候 QPS 可能是 "FAIL"
                        l_val = parts[0]
                        qps = parts[1]
                        lat = parts[2]
                        recall = parts[3]
                        
                        results.append([
                            metadata['dataset'],
                            metadata['R'],
                            metadata['alpha_min'],
                            metadata['alpha_max'],
                            current_algo,
                            l_val,
                            qps,
                            lat,
                            recall
                        ])
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        
    return results

def main():
    # 准备 CSV Header
    header = ['Dataset', 'R', 'Alpha_Min', 'Alpha_Max', 'Algorithm', 'L', 'QPS', 'Latency', 'Recall']
    
    all_data = []
    files = glob.glob(os.path.join(INPUT_DIR, "*.txt"))
    print(f"Found {len(files)} result files. Processing...")

    for filepath in files:
        # 1. 解析文件名参数
        metadata = parse_filename(filepath)
        if not metadata:
            print(f"[Skip] Invalid filename format: {filepath}")
            continue
            
        # 2. 解析文件内容
        file_data = parse_file_content(filepath, metadata)
        all_data.extend(file_data)

    # 3. 排序 (为了好看，按 Dataset -> R -> Min -> Max -> Algo -> L 排序)
    all_data.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4], int(x[5])))

    # 4. 写入 CSV
    with open(OUTPUT_CSV, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(all_data)
        
    print(f"========================================")
    print(f"Done! Summary saved to: {OUTPUT_CSV}")
    print(f"Total rows extracted: {len(all_data)}")
    print(f"========================================")

if __name__ == "__main__":
    main()