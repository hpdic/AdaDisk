import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import sys

# ==========================================
# 0. 配置
# ==========================================
OUTPUT_DIR = 'figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def find_file(filename):
    # 在常用目录里找 CSV
    search_paths = ['.', 'results', './results', '../experiments/scripts', '../experiments/results']
    for p in search_paths:
        path = os.path.join(p, filename)
        if os.path.exists(path): return path
    return None

CSV_PATH = find_file('speed_scan_summary.csv')

# 【严谨性检查】找不到文件直接报错，不瞎编
if not CSV_PATH:
    print("Error: 找不到 speed_scan_summary.csv！请检查文件路径。")
    print("代码已停止，未生成任何图片。")
    sys.exit(1)

print(f"正在读取真实数据: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)

# ==========================================
# 1. 绘图风格
# ==========================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 16,          
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 16,
    "legend.fontsize": 13,
    "hatch.linewidth": 1.0,
    "axes.grid": True,
    "grid.alpha": 0.4,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.constrained_layout.use": True
})

COLORS = {'MCGI': '#C82423', 'DiskANN': '#2878B5'}
LABELS = {'MCGI': 'MCGI', 'DiskANN': 'DiskANN'}

# ==========================================
# 2. 绘图逻辑
# ==========================================
def plot_bars(df, dataset='gist'):
    print(f"Processing dataset: {dataset}...")
    
    # --- 图 1: Sensitivity (L vs Recall) ---
    # 这里选点您可以根据真实数据的 L 分布微调
    target_Ls = [40, 80, 120, 160, 200] 
    
    mcgi_recalls = []
    diskann_recalls = []
    final_Ls = []

    for l_val in target_Ls:
        sub = df[df['Dataset'] == dataset]
        available_Ls = sub['L'].unique()
        if len(available_Ls) == 0: continue
        
        # 找最接近的 L
        nearest_L = available_Ls[np.argmin(np.abs(available_Ls - l_val))]
        
        if nearest_L in final_Ls: continue
        final_Ls.append(nearest_L)

        # 读取真实 Recall
        r_mcgi = sub[(sub['Algorithm'] == 'MCGI') & (sub['L'] == nearest_L)]['Recall'].max()
        r_base = sub[(sub['Algorithm'] == 'Baseline') & (sub['L'] == nearest_L)]['Recall'].max()
        
        # 数据清洗 (处理 NaN 和 百分比归一化)
        if pd.isna(r_mcgi): r_mcgi = 0
        if pd.isna(r_base): r_base = 0
        
        # 自动检测是否是 0-100 的数据，统一转为 0-1
        if df['Recall'].max() > 2.0: 
             if r_mcgi > 1.0: r_mcgi /= 100.0
             if r_base > 1.0: r_base /= 100.0
        
        mcgi_recalls.append(r_mcgi)
        diskann_recalls.append(r_base)

    # 画 Sensitivity 图
    plt.figure(figsize=(4, 4)) 
    x = np.arange(len(final_Ls))
    width = 0.35

    plt.bar(x - width/2, mcgi_recalls, width, label=LABELS['MCGI'], 
            color=COLORS['MCGI'], edgecolor='black', zorder=3)
    plt.bar(x + width/2, diskann_recalls, width, label=LABELS['DiskANN'], 
            color=COLORS['DiskANN'], edgecolor='black', hatch='///', zorder=3)

    plt.xlabel('List Size ($L$)') 
    plt.ylabel('Recall@10')
    plt.xticks(x, final_Ls)
    
    # 动态调整 Y 轴
    valid_vals = [v for v in diskann_recalls + mcgi_recalls if v > 0]
    min_val = min(valid_vals) if valid_vals else 0
    plt.ylim(max(0, min_val * 0.95), 1.01) 
    
    plt.legend(loc='upper left', framealpha=0.9, handlelength=1.5, handletextpad=0.5)
    plt.grid(axis='y', linestyle='--', alpha=0.4) 
    
    plt.savefig(f'{OUTPUT_DIR}/appendix_L_sensitivity.pdf')
    print(f"Saved {OUTPUT_DIR}/appendix_L_sensitivity.pdf")
    plt.close()

    # --- 图 2: Latency vs Recall Targets ---
    # 选取高 Recall 区域
    recall_targets = [0.92, 0.93, 0.94, 0.95, 0.96] 
    target_labels = [f"{int(t*100)}%" for t in recall_targets]
    
    mcgi_lats = []
    diskann_lats = []

    for t in recall_targets:
        sub = df[df['Dataset'] == dataset]
        
        # 归一化 Recall 列用于比较
        is_percent = sub['Recall'].max() > 2.0
        r_col = sub['Recall'] / 100.0 if is_percent else sub['Recall']

        # 找满足 recall 要求的最小 latency
        m_rows = sub[(sub['Algorithm'] == 'MCGI') & (r_col >= t)]
        lat_mcgi = m_rows['Latency'].min() if not m_rows.empty else np.nan
        
        d_rows = sub[(sub['Algorithm'] == 'Baseline') & (r_col >= t)]
        lat_base = d_rows['Latency'].min() if not d_rows.empty else np.nan
        
        # us -> ms
        mcgi_lats.append(lat_mcgi / 1000.0) 
        diskann_lats.append(lat_base / 1000.0) 

    # 画 Latency 图
    plt.figure(figsize=(4, 4))
    x = np.arange(len(recall_targets))
    width = 0.35

    plt.bar(x - width/2, mcgi_lats, width, label=LABELS['MCGI'], 
            color=COLORS['MCGI'], edgecolor='black', zorder=3)
    plt.bar(x + width/2, diskann_lats, width, label=LABELS['DiskANN'], 
            color=COLORS['DiskANN'], edgecolor='black', hatch='///', zorder=3)

    plt.xlabel('Target Recall')
    plt.ylabel('Latency (ms)')
    plt.yscale('log')
    plt.xticks(x, target_labels)
    
    plt.legend(loc='upper left', framealpha=0.9, handlelength=1.5, handletextpad=0.5)
    
    plt.grid(axis='y', linestyle='--', alpha=0.4, which='major')
    plt.grid(axis='y', linestyle=':', alpha=0.2, which='minor')

    plt.savefig(f'{OUTPUT_DIR}/appendix_latency.pdf')
    print(f"Saved {OUTPUT_DIR}/appendix_latency.pdf")
    plt.close()

if __name__ == "__main__":
    plot_bars(df, dataset='gist')