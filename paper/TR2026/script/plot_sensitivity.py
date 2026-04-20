import os
import matplotlib.pyplot as plt

# ================= Data Section =================
# Data from Group: R=48, min=1.1
baseline_recall = [79.3, 90.13, 93.97, 96.13]
baseline_qps = [1335.47, 736.09, 108.1, 57.71]

mcgi_12_recall = [77.59, 88.73, 93.21, 95.44]
mcgi_12_qps = [469.38, 233.78, 152.65, 134.19]

mcgi_15_recall = [79.36, 90.15, 94.36, 96.18]
mcgi_15_qps = [461.99, 252.01, 160.16, 130.19]

mcgi_17_recall = [79.48, 90.12, 94.50, 96.45]
mcgi_17_qps = [458.75, 251.98, 167.43, 131.16]

# ================= Style Configuration =================
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'legend.fontsize': 11,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'lines.linewidth': 2.2,
    'lines.markersize': 8
})

plt.figure(figsize=(8, 6))

# ================= Plotting =================
# Baseline
plt.plot(baseline_recall, baseline_qps, marker='o', linestyle='--', color='#555555', label='Baseline (DiskANN)')

# MCGI 1.2
plt.plot(mcgi_12_recall, mcgi_12_qps, marker='^', linestyle='-', color='#1f77b4', label='MCGI (max=1.2)')

# MCGI 1.5 (Added back)
plt.plot(mcgi_15_recall, mcgi_15_qps, marker='s', linestyle='-', color='#ff7f0e', label='MCGI (max=1.5)')

# MCGI 1.7 (Best performer)
plt.plot(mcgi_17_recall, mcgi_17_qps, marker='D', linestyle='-', linewidth=3.2, markersize=9, color='#d62728', label='MCGI (max=1.7)')

# ================= Axes and Grid =================
plt.xlabel('Recall@10 (%)')
plt.ylabel('Queries Per Second (QPS)')
plt.yscale('log')
plt.xlim(85, 97)

plt.grid(True, which='major', linestyle='-', alpha=0.5, color='gray')
plt.grid(True, which='minor', linestyle=':', alpha=0.3, color='gray')

# ================= Legend and Save =================
plt.legend(loc='upper right', framealpha=0.9, edgecolor='black')
plt.tight_layout()

output_dir = os.path.expanduser('~/hpdic/AdaDisk/paper/TR2026/figures/')
os.makedirs(output_dir, exist_ok=True)
output_filename = os.path.join(output_dir, 'ablation_qps_vs_recall.pdf')
plt.savefig(output_filename, format='pdf', bbox_inches='tight')
print('Chart generated and saved to: ' + output_filename)