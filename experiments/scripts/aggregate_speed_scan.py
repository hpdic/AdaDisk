import os
import re
import csv
import glob

# ================= 配置区 =================
# 输入文件夹 (根据昨天的脚本，文件夹应该是 speed_scan_85_95)
# 如果你有多个文件夹想合并，可以把它们都放在一个列表里，或者手动修改这里
INPUT_DIRS = [
    "experiments/speed_scan_85_95",
]

# 输出文件路径
OUTPUT_CSV = "experiments/speed_scan_summary.csv"

def parse_filename(filename):
    """
    从文件名解析参数。
    兼容两种格式：
    1. 新格式 (带L): sift_R32_L50_min1.0_max1.1.txt
    2. 旧格式 (不带L): gist_R32_min1.0_max1.1.txt
    """
    basename = os.path.basename(filename)
    
    # 1. 尝试匹配新格式 (带 _L)
    # Regex: [Dataset]_R[R]_L[L]_min[Min]_max[Max].txt
    match_new = re.search(r'([a-zA-Z0-9]+)_R(\d+)_L(\d+)_min([\d\.]+)_max([\d\.]+)\.txt', basename)
    if match_new:
        return {
            'dataset': match_new.group(1),
            'R': int(match_new.group(2)),
            # 注意：虽然文件名里有L，但我们通常还是信赖文件内容里的L (parts[0])
            # 不过这里记录一下也没坏处，作为元数据
            'L_filename': int(match_new.group(3)), 
            'alpha_min': float(match_new.group(4)),
            'alpha_max': float(match_new.group(5))
        }

    # 2. 尝试匹配旧格式 (不带 _L)
    # Regex: [Dataset]_R[R]_min[Min]_max[Max].txt
    match_old = re.search(r'([a-zA-Z0-9]+)_R(\d+)_min([\d\.]+)_max([\d\.]+)\.txt', basename)
    if match_old:
        return {
            'dataset': match_old.group(1),
            'R': int(match_old.group(2)),
            'L_filename': None, # 旧格式文件名不含L
            'alpha_min': float(match_old.group(3)),
            'alpha_max': float(match_old.group(4))
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
                if current_algo and re.match(r'^\d+', line):
                    parts = line.split()
                    if len(parts) >= 4:
                        l_val = parts[0]
                        qps = parts[1]
                        lat = parts[2]
                        recall = parts[3]
                        
                        # 如果 QPS 是 "FAIL" 或其他非数字，可以选择跳过或记录
                        try:
                            float(qps)
                        except ValueError:
                            continue

                        results.append([
                            metadata['dataset'],
                            metadata['R'],
                            metadata['alpha_min'],
                            metadata['alpha_max'],
                            current_algo,
                            l_val,   # 使用文件内容里的实际 L
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
    
    # 收集所有输入目录下的 txt 文件
    files = []
    for d in INPUT_DIRS:
        if os.path.exists(d):
            found = glob.glob(os.path.join(d, "*.txt"))
            print(f"Directory '{d}': found {len(found)} files.")
            files.extend(found)
        else:
            print(f"Directory '{d}' does not exist, skipping.")

    if not files:
        print("No files found via check paths. Please check INPUT_DIRS.")
        return

    print(f"Processing {len(files)} files total...")

    for filepath in files:
        # 1. 解析文件名参数
        metadata = parse_filename(filepath)
        if not metadata:
            print(f"[Skip] Invalid filename format: {os.path.basename(filepath)}")
            continue
            
        # 2. 解析文件内容
        file_data = parse_file_content(filepath, metadata)
        all_data.extend(file_data)

    # 3. 排序 (按 Dataset -> R -> Min -> Max -> Algo -> L 排序)
    # x[5] is L, convert to int for correct numerical sorting
    all_data.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4], int(x[5])))

    # 4. 写入 CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
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