import matplotlib.pyplot as plt
import numpy as np
import os

# ================= 0. 路径配置 =================
OUTPUT_DIR = os.path.expanduser("~/hpdic/AdaDisk/paper/TR2026/figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "t2i_performance.pdf")

# ================= 1. 数据准备 (T2I-10M New Run) =================
# --- DiskANN (Baseline) ---
# 保持不变
base_recall = [
    57.46, 70.00, 80.30, 88.09, 90.02, 
    91.42, 92.48, 93.34, 94.03, 94.60
]
base_qps = [
    43292.75, 27795.34, 16534.32, 9133.37, 7448.96, 
    6283.00, 5446.46, 4799.23, 4315.48, 3899.28
]
base_latency_ms = [
    2.18, 3.41, 5.77, 10.46, 12.83, 
    15.21, 17.56, 19.93, 22.17, 24.54
]

# --- MCGI (Ours) - New Data ---
# 数据来源：User Log (L=10 to 200)
# Latency 单位已转为 ms (/1000)
mcgi_recall = [
    56.68, 69.60, 79.99, 87.87, 89.91, 
    91.39, 92.48, 93.33, 94.00, 94.57
]
mcgi_qps = [
    43986.03, 28409.03, 16985.90, 9481.81, 7730.43, 
    6592.73, 5704.39, 5035.36, 4439.65, 4014.05
]
mcgi_latency_ms = [
    2.15, 3.34, 5.61, 10.08, 12.36, 
    14.50, 16.76, 19.00, 21.55, 23.84
]

# ================= 2. 画图配置 (视觉聚焦策略) =================
plt.rcParams.update({
    'font.size': 26,            
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'], 
    'lines.linewidth': 4,       
    'lines.markersize': 14,     
    'axes.grid': True,
    'grid.alpha': 0.3,
    'legend.fontsize': 22,      
    'axes.labelsize': 28,       
    'xtick.labelsize': 24,      
    'ytick.labelsize': 24,
    'figure.autolayout': False
})

fig, axes = plt.subplots(1, 2, figsize=(14, 10))

# --- 左图: QPS vs Recall ---
ax1 = axes[0]
ax1.plot(base_recall, base_qps, 'o-', color='#1f77b4', label='DiskANN', zorder=1)
ax1.plot(mcgi_recall, mcgi_qps, 's--', color='#d62728', label='MCGI (Ours)', zorder=2)

ax1.set_xlabel('Recall@10 (%)')
ax1.set_ylabel('QPS (Queries/s)')

# 【聚焦高精度区域】
# L<80 的点 (Recall ~87.8% 及以下) 会被自动切掉，符合"忽略低 Recall"的需求
ax1.set_xlim(88, 95.5)
ax1.set_ylim(3000, 10000) 

ax1.legend(loc='upper right', frameon=True, edgecolor='black', framealpha=0.9)
ax1.grid(True, linestyle='--', which='both')
ax1.set_title("(a) Throughput (High Recall)", y=-0.27, fontsize=28) 

# --- 右图: Latency vs Recall ---
ax2 = axes[1]
ax2.plot(base_recall, base_latency_ms, 'o-', color='#1f77b4', label='DiskANN', zorder=1)
ax2.plot(mcgi_recall, mcgi_latency_ms, 's--', color='#d62728', label='MCGI (Ours)', zorder=2)

ax2.set_xlabel('Recall@10 (%)')
ax2.set_ylabel('Latency (ms)')

# 【聚焦高精度区域】
ax2.set_xlim(88, 95.5)
ax2.set_ylim(10, 26)

ax2.legend(loc='upper left', frameon=True, edgecolor='black', framealpha=0.9)
ax2.grid(True, linestyle='--', which='both')
ax2.set_title("(b) Latency (High Recall)", y=-0.27, fontsize=28) 

plt.tight_layout()
plt.subplots_adjust(bottom=0.23, wspace=0.35) 

plt.savefig(OUTPUT_FILE, format='pdf', bbox_inches='tight')
print(f"✅ New T2I Figure Saved to: {OUTPUT_FILE}")