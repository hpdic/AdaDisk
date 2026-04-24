import matplotlib.pyplot as plt
import numpy as np
import os

# ================= 0. 路径配置 =================
OUTPUT_DIR = os.path.expanduser('~/hpdic/AdaDisk/paper/TR2026/figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'sift1b_performance.pdf')

# ================= 1. 数据准备 (SIFT1B 10亿数据) =================
# 1.1 DiskANN (Baseline)
base_recall = [
    45.94, 58.16, 69.87, 79.79, 82.58, 
    84.66, 86.24, 87.51, 88.63, 89.56
]
base_qps = [
    21653.28, 15693.77, 10065.47, 5874.33, 4874.16, 
    4135.08, 3610.91, 3181.28, 2867.43, 2597.32
]
base_latency_ms = [
    5.81, 8.08, 12.62, 21.66, 26.12, 
    30.79, 35.26, 40.04, 44.43, 49.06
]

# 1.2 MCGI (Ours)
mcgi_recall = [
    59.81, 71.59, 81.50, 84.22, 
    86.23, 87.76, 89.00, 90.02, 90.84
]
mcgi_qps = [
    12127.03, 8036.93, 5121.02, 5455.30, 
    4773.02, 4238.72, 3794.81, 3436.17, 3126.13
]
mcgi_latency_ms = [
    4.56, 6.91, 10.86, 10.20, 
    11.65, 13.12, 14.66, 16.20, 17.80
]

# 1.3 PipeANN (New Baseline, L=100 to 500)
pipe_recall = [
    75.30, 83.78, 87.72, 90.12, 91.77
]
pipe_qps = [
    3828.11, 2865.44, 2181.98, 1750.87, 1527.03
]
pipe_latency_ms = [
    24.52, 32.89, 43.23, 53.77, 61.80
]

# ================= 2. 画图配置 (一致性修正版) =================
plt.rcParams.update({
    'font.size': 26,            
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'], 
    'lines.linewidth': 4,       
    'lines.markersize': 14,     
    'axes.grid': True,
    'grid.alpha': 0.3,
    'legend.fontsize': 24,      
    'axes.labelsize': 28,       
    'xtick.labelsize': 24,      
    'ytick.labelsize': 24,
    'figure.autolayout': False
})

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

color_mcgi = '#800080'   # 紫色
color_pipe = '#D4AF37'   # 金色
color_disk = '#2CA02C'   # 绿色

# 2.1 左图: QPS vs Recall
ax1 = axes[0]
ax1.plot(base_recall, base_qps, 'o-', color=color_disk, label='DiskANN', zorder=1)
ax1.plot(pipe_recall, pipe_qps, 'D-.', color=color_pipe, label='PipeANN', zorder=2)
ax1.plot(mcgi_recall, mcgi_qps, 's--', color=color_mcgi, label='MCGI', zorder=3)

ax1.set_xlabel('Recall@10 (%)')
ax1.set_ylabel('QPS (Queries/s)')

# --- 核心改动：精准截断 X 轴并锁定 Y 轴 ---
ax1.set_xlim(70, 93) 
ax1.set_ylim(1000, 10500) # 根据 70% 后的最高 QPS 重新定高

ax1.legend(loc='upper right', frameon=True, edgecolor='black', framealpha=0.8)
ax1.grid(True, linestyle='--', which='both')
ax1.set_title('(a) Throughput (QPS)', y=-0.32, fontsize=28) 

# 2.2 右图: Latency vs Recall
ax2 = axes[1]
ax2.plot(base_recall, base_latency_ms, 'o-', color=color_disk, label='DiskANN', zorder=1)
ax2.plot(pipe_recall, pipe_latency_ms, 'D-.', color=color_pipe, label='PipeANN', zorder=2)
ax2.plot(mcgi_recall, mcgi_latency_ms, 's--', color=color_mcgi, label='MCGI', zorder=3)

ax2.set_xlabel('Recall@10 (%)')
ax2.set_ylabel('Latency (ms)')

# --- 核心改动：精准截断 X 轴并锁定 Y 轴 ---
ax2.set_xlim(70, 93)
ax2.set_ylim(0, 65) # 根据最高延迟 61.8ms 重新定高

ax2.legend(loc='upper left', frameon=True, edgecolor='black', framealpha=0.8)
ax2.grid(True, linestyle='--', which='both')
ax2.set_title('(b) Latency', y=-0.32, fontsize=28) 

plt.tight_layout()
plt.subplots_adjust(bottom=0.25, wspace=0.35) 

plt.savefig(OUTPUT_FILE, format='pdf', bbox_inches='tight')
print(f'Figure Saved to: {OUTPUT_FILE}')