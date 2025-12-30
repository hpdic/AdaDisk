import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.ticker as ticker

# ==========================================
# 0. 自动寻路与配置
# ==========================================
OUTPUT_DIR = 'figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 搜索路径：当前目录 -> results 子目录 -> 上级目录
POSSIBLE_PATHS = ['.', 'results', './results', '../experiments/scripts']

def find_file(filename):
    for p in POSSIBLE_PATHS:
        full_path = os.path.join(p, filename)
        if os.path.exists(full_path):
            print(f"[Info] Found {filename} at: {os.path.abspath(full_path)}")
            return full_path
    return None

CSV_PATH = find_file('speed_scan_summary.csv')
TXT_PATH = find_file('result_faiss.txt')

if not CSV_PATH or not TXT_PATH:
    print("\n[Error] 找不到数据文件！请确保文件在当前目录或 results 文件夹中。")
    exit(1)

# ==========================================
# 1. 精致的绘图风格 (Academic Style)
# ==========================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"], # 优先用 Times
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "lines.linewidth": 2.5,    # 调细一点，更精致
    "lines.markersize": 8,     # 调小一点，不显得拥挤
    "axes.grid": True,
    "grid.alpha": 0.3,         # 网格淡一点
    "grid.linestyle": "--",
    "pdf.fonttype": 42,        # 嵌入字体
    "ps.fonttype": 42,
    "figure.autolayout": True  # 自动布局防遮挡
})

# 经典学术配色 (Set1 / Tableau 10 变体)
COLORS = {'MCGI': '#C82423', 'DiskANN': '#2878B5', 'Faiss': '#9AC9DB'} 
MARKERS = {'MCGI': 'o', 'DiskANN': 's', 'Faiss': '^'}
STYLES  = {'MCGI': '-', 'DiskANN': '--', 'Faiss': '-.'}

# ==========================================
# 2. 帕累托前沿 (Pareto Frontier)
# ==========================================
def get_pareto_frontier(recall, qps):
    """只保留最优边界点，让曲线平滑单调"""
    if len(recall) == 0: return [], []
    # 1. 排序
    points = sorted(zip(recall, qps), key=lambda x: x[0])
    
    # 2. 从右向左筛选 (保留更高 QPS 的点)
    pareto_points = []
    max_qps = -1.0
    for r, q in reversed(points):
        if q >= max_qps:
            pareto_points.append((r, q))
            max_qps = q
            
    # 3. 翻转回来
    pareto_points.reverse()
    return [p[0] for p in pareto_points], [p[1] for p in pareto_points]

# ==========================================
# 3. 数据解析
# ==========================================
def parse_faiss_log(filename):
    data = {'sift': {'r': [], 'q': []}, 'gist': {'r': [], 'q': []}, 'glove': {'r': [], 'q': []}}
    curr_data = None
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if "=== Benchmark Result:" in line:
                    if "SIFT" in line: curr_data = data['sift']
                    elif "GIST" in line: curr_data = data['gist']
                    elif "GLOVE" in line: curr_data = data['glove']
                elif "|" in line and "Recall" not in line and curr_data is not None:
                    parts = line.split('|')
                    if len(parts) >= 4:
                        try:
                            r = float(parts[2].strip())
                            q = float(parts[3].strip())
                            curr_data['r'].append(r)
                            curr_data['q'].append(q)
                        except: pass
    except Exception as e:
        print(f"[Warning] Error parsing Faiss log: {e}")
    return data

# ==========================================
# 4. 核心绘图逻辑
# ==========================================
def plot_dataset(dataset_key, title_name, df, faiss_data, x_min=None, log_scale=False):
    print(f"Drawing {title_name}...")
    fig, ax = plt.subplots(figsize=(6, 5)) # 标准单栏尺寸

    # 1. MCGI
    sub_df = df[(df['Dataset'] == dataset_key) & (df['Algorithm'] == 'MCGI')]
    if not sub_df.empty:
        r = sub_df['Recall'].values
        if r.max() > 1.0: r = r / 100.0
        q = sub_df['QPS'].values
        pr, pq = get_pareto_frontier(r, q)
        ax.plot(pr, pq, label='MCGI (Ours)', color=COLORS['MCGI'], 
                marker=MARKERS['MCGI'], linestyle=STYLES['MCGI'], zorder=10)

    # 2. DiskANN
    sub_df = df[(df['Dataset'] == dataset_key) & (df['Algorithm'] == 'Baseline')]
    if not sub_df.empty:
        r = sub_df['Recall'].values
        if r.max() > 1.0: r = r / 100.0
        q = sub_df['QPS'].values
        pr, pq = get_pareto_frontier(r, q)
        ax.plot(pr, pq, label='DiskANN', color=COLORS['DiskANN'], 
                marker=MARKERS['DiskANN'], linestyle=STYLES['DiskANN'], zorder=5)

    # 3. Faiss
    fd = faiss_data.get(dataset_key)
    if fd and fd['r']:
        pr, pq = get_pareto_frontier(fd['r'], fd['q'])
        ax.plot(pr, pq, label='Faiss (IVF)', color=COLORS['Faiss'], 
                marker=MARKERS['Faiss'], linestyle=STYLES['Faiss'], zorder=1)

    # 样式微调
    ax.set_xlabel(f'Recall@10')
    ax.set_ylabel('QPS (Queries/s)')
    
    # 标题 (可选，这里加上更清楚，Paper里可以通过 LaTeX caption 覆盖)
    # ax.set_title(title_name) 

    # GIST 特殊处理
    if log_scale:
        ax.set_yscale('log')
        # GIST 允许显示低 Recall 区域以展示全貌
        if x_min is None: x_min = 0.1
    else:
        # SIFT/GloVe 聚焦高 Recall
        if x_min is None: x_min = 0.85
        
    ax.set_xlim(left=x_min, right=1.01)
    
    # 图例优化：放在“最佳”位置，半透明背景
    ax.legend(loc='best', frameon=True, fancybox=False, framealpha=0.8, edgecolor='gray')
    
    # 保存
    out_path = os.path.join(OUTPUT_DIR, f"{dataset_key}_recall_qps.pdf")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f" -> Saved to {out_path}")
    plt.close()

# ==========================================
# 5. 执行
# ==========================================
if __name__ == "__main__":
    # 加载数据
    df = pd.read_csv(CSV_PATH)
    faiss_data = parse_faiss_log(TXT_PATH)

    # 绘制三张图
    # GIST: 开启 Log 轴，范围放宽一点
    plot_dataset('gist', 'GIST1M', df, faiss_data, x_min=0.1, log_scale=True)
    
    # SIFT: 聚焦 0.9+
    plot_dataset('sift', 'SIFT1M', df, faiss_data, x_min=0.90, log_scale=False)
    
    # GloVe: 聚焦 0.85+
    plot_dataset('glove', 'GloVe-100', df, faiss_data, x_min=0.85, log_scale=False)

    print("\n✅ 所有图片已生成完毕！请查看 figures 文件夹。")