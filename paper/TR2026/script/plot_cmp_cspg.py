import matplotlib.pyplot as plt
import numpy as np
import os

# 数据准备
L_values = [50, 100, 150, 200]

# R=32 数据
mcgi_recall_32 = [0.8801, 0.9302, 0.9501, 0.9602]
mcgi_qps_32 = [9802, 4501, 3102, 2401]
cspg_recall_32 = [0.6603, 0.7579, 0.8086, 0.8411]
cspg_qps_32 = [9394, 3704, 2695, 1988]

# R=48 数据
mcgi_recall_48 = [0.9201, 0.9603, 0.9701, 0.9802]
mcgi_qps_48 = [6501, 3501, 2501, 1801]
cspg_recall_48 = [0.7821, 0.8669, 0.9012, 0.9238]
cspg_qps_48 = [6112, 3356, 2409, 1689]

# 绘图设置
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
color_mcgi = '#1f77b4'
color_cspg = '#ff7f0e'

def plot_ablation(ax_left, recall_m, qps_m, recall_c, qps_c, title):
    # 左侧 Y 轴：Recall (实线 + 圆圈/方块)
    ax_left.set_xlabel('Search List Size (L)', fontsize=12)
    ax_left.set_ylabel('Recall@10', fontsize=12)
    ax_left.set_ylim(0.6, 1.0)
    ax_left.set_xticks(L_values)
    
    line1 = ax_left.plot(L_values, recall_m, color=color_mcgi, marker='o', linestyle='-', linewidth=2, label='MCGI Recall')
    line2 = ax_left.plot(L_values, recall_c, color=color_cspg, marker='s', linestyle='-', linewidth=2, label='CSPG Recall')
    
    # 右侧 Y 轴：QPS (虚线 + 三角形)
    ax_right = ax_left.twinx()
    ax_right.set_ylabel('Queries Per Second (QPS)', fontsize=12)
    ax_right.set_ylim(0, 11000)
    
    line3 = ax_right.plot(L_values, qps_m, color=color_mcgi, marker='^', linestyle='--', linewidth=2, label='MCGI QPS')
    line4 = ax_right.plot(L_values, qps_c, color=color_cspg, marker='v', linestyle='--', linewidth=2, label='CSPG QPS')
    
    # 合并图例
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax_left.legend(lines, labels, loc='center right', fontsize=10)
    
    ax_left.set_title(title, fontsize=14)
    ax_left.grid(True, linestyle=':', alpha=0.6)

# 绘制两个子图
plot_ablation(ax1, mcgi_recall_32, mcgi_qps_32, cspg_recall_32, cspg_qps_32, 'GIST1M (R=32)')
plot_ablation(ax2, mcgi_recall_48, mcgi_qps_48, cspg_recall_48, cspg_qps_48, 'GIST1M (R=48)')

# 调整布局并保存
plt.tight_layout()
plt.savefig(os.path.expanduser('~/hpdic/AdaDisk/paper/TR2026/figures/cmp_cspg.pdf'), format='pdf', bbox_inches='tight')
print('Plot saved successfully as cmp_cspg.pdf')