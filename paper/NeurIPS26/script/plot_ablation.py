import os
import matplotlib.pyplot as plt

# ================= 数据配置区 =================

# 1. Sensitivity Analysis (R=48, min=1.1)
s_recall = [79.3, 90.13, 93.97, 96.13]
s_qps_base = [1335.47, 736.09, 108.1, 57.71]
s_qps_mcgi_12 = [469.38, 233.78, 152.65, 134.19]
s_qps_mcgi_15 = [461.99, 252.01, 160.16, 130.19]
s_qps_mcgi_17 = [458.75, 251.98, 167.43, 131.16]

# 2. Ablation Group 1: R=48, min=1.1, max=1.7
# Sigmoid 数据来自你的完整日志
a1_recall_sig = [79.48, 90.12, 94.50, 96.45]
a1_qps_sig = [458.75, 251.98, 167.43, 131.16]
# Linear 数据：请从 run_ablation.sh 跑出的第一组结果中获取并替换
a1_recall_lin = [78.5, 89.2, 93.1, 95.2] 
a1_qps_lin = [400, 200, 100, 80]

# 3. Ablation Group 2: R=32, min=1.0, max=2.0
# Sigmoid 数据来自你的完整日志
a2_recall_sig = [74.61, 86.54, 91.03, 93.39]
a2_qps_sig = [438.23, 256.04, 164.86, 136.61]
# Linear 数据：请从 run_ablation.sh 跑出的第二组结果中获取并替换
a2_recall_lin = [73.5, 85.1, 89.5, 92.1]
a2_qps_lin = [380, 210, 110, 90]

# 4. Ablation Group 3: R=32, min=1.1, max=2.0
# Sigmoid 数据来自你的完整日志
a3_recall_sig = [74.83, 86.31, 91.02, 93.32]
a3_qps_sig = [458.89, 252.98, 157.87, 140.02]
# Linear 数据：请从 run_ablation.sh 跑出的第三组结果中获取并替换
a3_recall_lin = [73.8, 85.4, 89.8, 92.4]
a3_qps_lin = [400, 220, 120, 95]

# 公共 Baseline (R=32 和 R=48 的 Recall 不同)
base_qps_32 = [1313.42, 731.72, 113.41, 58.95]
base_rec_32 = [74.98, 86.13, 90.52, 93.04]
base_qps_48 = [1335.47, 736.09, 108.1, 57.71]
base_rec_48 = [79.3, 90.13, 93.97, 96.13]

# ================= 绘图样式 =================
plt.rcParams.update({
    'font.size': 18,
    'axes.labelsize': 20,
    'legend.fontsize': 14,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'axes.titlesize': 18,
    'lines.linewidth': 2.5,
    'lines.markersize': 8
})

fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))

def plot_common(ax, title):
    ax.set_yscale('log')
    ax.set_xlabel('Recall@10 (%)')
    ax.grid(True, which='both', linestyle=':', alpha=0.5)
    ax.set_title(title)

# Panel 1: Sensitivity
ax = axes[0]
ax.plot(base_rec_48, base_qps_48, 'o--', color='#555555', label='Baseline')
ax.plot(base_rec_48, s_qps_mcgi_12, '^--', color='#1f77b4', label='max=1.2')
ax.plot(base_rec_48, s_qps_mcgi_15, 's--', color='#ff7f0e', label='max=1.5')
ax.plot(base_rec_48, s_qps_mcgi_17, 'D-', color='#d62728', label='max=1.7')
ax.set_xlim(85, 97)
plot_common(ax, '(a) Sensitivity Analysis')
ax.set_ylabel('QPS (log scale)')
ax.legend(loc='lower left')

# Panel 2: Ablation R=48
ax = axes[1]
ax.plot(base_rec_48, base_qps_48, 'o--', color='#555555', label='Baseline')
ax.plot(a1_recall_sig, a1_qps_sig, 'D-', color='#d62728', label='Sigmoid')
ax.plot(a1_recall_lin, a1_qps_lin, 'x-', color='#9467bd', label='Linear')
ax.set_xlim(85, 97)
plot_common(ax, '(b) Ablation: R=48, max=1.7')
ax.legend(loc='lower left')

# Panel 3: Ablation R=32 (min=1.0)
ax = axes[2]
ax.plot(base_rec_32, base_qps_32, 'o--', color='#555555', label='Baseline')
ax.plot(a2_recall_sig, a2_qps_sig, 'D-', color='#d62728', label='Sigmoid')
ax.plot(a2_recall_lin, a2_qps_lin, 'x-', color='#9467bd', label='Linear')
ax.set_xlim(85, 95)
plot_common(ax, '(c) Ablation: R=32, min=1.0')
ax.legend(loc='lower left')

# Panel 4: Ablation R=32 (min=1.1)
ax = axes[3]
ax.plot(base_rec_32, base_qps_32, 'o--', color='#555555', label='Baseline')
ax.plot(a3_recall_sig, a3_qps_sig, 'D-', color='#d62728', label='Sigmoid')
ax.plot(a3_recall_lin, a3_qps_lin, 'x-', color='#9467bd', label='Linear')
ax.set_xlim(85, 95)
plot_common(ax, '(d) Ablation: R=32, min=1.1')
ax.legend(loc='lower left')

plt.tight_layout()
output_path = os.path.expanduser('~/hpdic/AdaDisk/paper/TR2026/figures/combined_ablation.pdf')
plt.savefig(output_path, format='pdf', bbox_inches='tight')
print('Combined figure saved to: ' + output_path)