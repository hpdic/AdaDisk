import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# ==========================================
# 0. 配置
# ==========================================
OUTPUT_DIR = 'figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 自动寻找数据
def find_file(filename):
    for p in ['.', 'results', './results', '../experiments/scripts']:
        path = os.path.join(p, filename)
        if os.path.exists(path): return path
    return None

CSV_PATH = find_file('speed_scan_summary.csv')
if not CSV_PATH:
    print("Error: 找不到 speed_scan_summary.csv")
    exit(1)

# ==========================================
# 1. 绘图风格 (保持论文一致性)
# ==========================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "lines.linewidth": 2.5,
    "lines.markersize": 8,
    "grid.alpha": 0.3,
    "axes.grid": True,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

COLORS = {'MCGI': '#C82423', 'DiskANN': '#2878B5'}
MARKERS = {'MCGI': 'o', 'DiskANN': 's'}

# ==========================================
# 2. 绘图逻辑
# ==========================================
def plot_analysis(df, dataset='gist'):
    print(f"Processing {dataset}...")
    
    # --- 图 1: L vs Recall (参数敏感性) ---
    plt.figure(figsize=(6, 5))
    for algo in ['MCGI', 'Baseline']:
        key = 'MCGI' if algo == 'MCGI' else 'DiskANN'
        sub = df[(df['Dataset'] == dataset) & (df['Algorithm'] == algo)]
        if sub.empty: continue
        
        # 取每个 L 的最佳 Recall
        best = sub.groupby('L')['Recall'].max().reset_index().sort_values('L')
        r = best['Recall'].values
        if r.max() > 1.0: r /= 100.0
        
        plt.plot(best['L'], r, label=f'{key}', color=COLORS[key], 
                 marker=MARKERS[key], linewidth=2.5)
                 
    plt.xlabel('Search List Size ($L$)')
    plt.ylabel('Recall@10')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/appendix_L_sensitivity.pdf')
    plt.close()

    # --- 图 2: Latency vs Recall (延迟分析) ---
    plt.figure(figsize=(6, 5))
    for algo in ['MCGI', 'Baseline']:
        key = 'MCGI' if algo == 'MCGI' else 'DiskANN'
        sub = df[(df['Dataset'] == dataset) & (df['Algorithm'] == algo)]
        if sub.empty: continue
        
        sub = sub.sort_values('Recall')
        r = sub['Recall'].values
        if r.max() > 1.0: r /= 100.0
        # Latency 通常是 us，转 ms
        lat = sub['Latency'].values / 1000.0 
        
        plt.plot(r, lat, label=f'{key}', color=COLORS[key], 
                 marker=MARKERS[key], linewidth=2.5)

    plt.xlabel('Recall@10')
    plt.ylabel('Latency (ms)')
    plt.yscale('log') # Log 轴展示长尾差异
    plt.legend(loc='upper left')
    plt.grid(True, which="minor", ls=":", alpha=0.2)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/appendix_latency.pdf')
    plt.close()

if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    plot_analysis(df, dataset='gist') # 只画 GIST