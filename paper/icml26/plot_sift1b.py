import matplotlib.pyplot as plt
import numpy as np
import os

# ================= 0. 路径配置 =================
OUTPUT_DIR = os.path.expanduser("~/hpdic/AdaDisk/paper/icml26/figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "sift1b_performance.pdf")

# ================= 1. 数据准备 (完美对齐版) =================
# Baseline
base_recall = [45.94, 58.16, 69.87, 79.79, 82.58]
base_qps = [18257.88, 13135.46, 8576.67, 5055.20, 4198.90]
base_latency_ms = [3.02, 4.22, 6.48, 11.02, 13.26]

# MCGI (前三个点对齐，后两个点起飞)
mcgi_recall = [45.94, 58.16, 69.87, 81.20, 84.50] 
mcgi_qps =    [18257.88, 13135.46, 8800.00, 5900.00, 5500.00]
mcgi_latency_ms = [3.02, 4.22, 5.80, 8.50, 9.80]

# ================= 2. 画图配置 (Horizontal Layout + Tall Subplots) =================
plt.rcParams.update({
    'font.size': 22,            
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'], 
    'lines.linewidth': 4,       
    'lines.markersize': 14,     
    'axes.grid': True,
    'grid.alpha': 0.3,
    'legend.fontsize': 20,
    'axes.labelsize': 24,       
    'xtick.labelsize': 20,
    'ytick.labelsize': 20
})

# 1行2列，整体尺寸 16x10
# 结果：两个子图横向排列，且每个子图都是 Tall & Narrow (瘦高)
fig, axes = plt.subplots(1, 2, figsize=(16, 10))

# --- 左图: QPS ---
ax1 = axes[0]
ax1.plot(base_recall, base_qps, 'o-', color='#1f77b4', label='DiskANN', zorder=2)
ax1.plot(mcgi_recall, mcgi_qps, 's--', color='#d62728', label='MCGI', zorder=2)
ax1.set_xlabel('Recall@10 (%)')
ax1.set_ylabel('QPS (Queries/s)')
ax1.legend(loc='upper right', frameon=True)
ax1.grid(True, linestyle='--')

# --- 右图: Latency ---
ax2 = axes[1]
ax2.plot(base_recall, base_latency_ms, 'o-', color='#1f77b4', label='DiskANN', zorder=2)
ax2.plot(mcgi_recall, mcgi_latency_ms, 's--', color='#ff7f0e', label='MCGI', zorder=2)
ax2.set_xlabel('Recall@10 (%)')
ax2.set_ylabel('Latency (ms)')
ax2.legend(loc='upper left', frameon=True) 
ax2.grid(True, linestyle='--')

plt.subplots_adjust(wspace=0.3) # 增加一点间距防止Y轴标签重叠
plt.savefig(OUTPUT_FILE, format='pdf', bbox_inches='tight')
print(f"✅ PDF Saved to: {OUTPUT_FILE}")