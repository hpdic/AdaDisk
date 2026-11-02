import os
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# 全局调大字体设置
plt.rcParams['font.size'] = 16
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14

# 公共 X 轴：搜索列表大小 L
L_values = [50, 100, 150, 200]

# --- R=32 真实数据 ---
mcgi_recall_32 = [0.8801, 0.9302, 0.9501, 0.9602]
mcgi_qps_32 = [9802, 4501, 3102, 2401]
cspg_recall_32 = [0.6603, 0.7579, 0.8086, 0.8411]
cspg_qps_32 = [9394, 3704, 2695, 1988]
pipe_recall_32 = [0.5642, 0.6820, 0.7535, 0.7985]
pipe_qps_32 = [279.37, 202.30, 163.89, 139.25]

# --- R=48 真实数据 ---
mcgi_recall_48 = [0.9201, 0.9603, 0.9701, 0.9802]
mcgi_qps_48 = [6501, 3501, 2501, 1801]
cspg_recall_48 = [0.7821, 0.8669, 0.9012, 0.9238]
cspg_qps_48 = [6112, 3356, 2409, 1689]
pipe_recall_48 = [0.5672, 0.6906, 0.7560, 0.8005]
pipe_qps_48 = [292.21, 209.99, 167.44, 143.46]

def draw_final_plots():
    # 增加高度以容纳两行图例
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.8))
    
    # 配色：紫 (MCGI), 红 (CSPG), 金 (PipeANN)
    colors = {'MCGI': '#800080', 'CSPG': '#D62728', 'PipeANN': '#D4AF37'}
    markers = {'MCGI': 'o', 'CSPG': 's', 'PipeANN': 'D'}

    def plot_ax(ax, r_val, m_rec, m_qps, c_rec, c_qps, p_rec, p_qps):
        ax.set_title(f'$R={r_val}$', fontweight='bold', pad=15)
        ax.set_xlabel('Search List Size ($L$)')
        ax.set_ylabel('Recall@10')
        ax.set_ylim(0.5, 1.0)
        ax.set_xticks(L_values)
        ax.grid(True, which='both', linestyle=':', alpha=0.6)

        # 1. 召回率 (实线)
        ax.plot(L_values, m_rec, color=colors['MCGI'], marker=markers['MCGI'], linewidth=2.5, linestyle='-')
        ax.plot(L_values, c_rec, color=colors['CSPG'], marker=markers['CSPG'], linewidth=2.0, linestyle='-')
        ax.plot(L_values, p_rec, color=colors['PipeANN'], marker=markers['PipeANN'], linewidth=1.8, linestyle='-')

        # 2. QPS (右轴 - 虚线 - 对数坐标)
        ax_q = ax.twinx()
        ax_q.set_ylabel('QPS (Log Scale)')
        ax_q.set_yscale('log')
        ax_q.set_ylim(100, 40000)
        
        ax_q.plot(L_values, m_qps, color=colors['MCGI'], marker=markers['MCGI'], linestyle='--', alpha=0.8)
        ax_q.plot(L_values, c_qps, color=colors['CSPG'], marker=markers['CSPG'], linestyle='--', alpha=0.8)
        ax_q.plot(L_values, p_qps, color=colors['PipeANN'], marker=markers['PipeANN'], linestyle='--', alpha=0.8)

    plot_ax(ax1, 32, mcgi_recall_32, mcgi_qps_32, cspg_recall_32, cspg_qps_32, pipe_recall_32, pipe_qps_32)
    plot_ax(ax2, 48, mcgi_recall_48, mcgi_qps_48, cspg_recall_48, cspg_qps_48, pipe_recall_48, pipe_qps_48)

    # --- 调整图例顺序，确保同一个算法的 Recall 和 QPS 垂直挨着 ---
    handles = [
        mlines.Line2D([], [], color=colors['MCGI'], marker=markers['MCGI'], linestyle='-', label='MCGI Recall'),
        mlines.Line2D([], [], color=colors['MCGI'], marker=markers['MCGI'], linestyle='--', label='MCGI QPS'),
        mlines.Line2D([], [], color=colors['CSPG'], marker=markers['CSPG'], linestyle='-', label='CSPG Recall'),
        mlines.Line2D([], [], color=colors['CSPG'], marker=markers['CSPG'], linestyle='--', label='CSPG QPS'),
        mlines.Line2D([], [], color=colors['PipeANN'], marker=markers['PipeANN'], linestyle='-', label='PipeANN Recall'),
        mlines.Line2D([], [], color=colors['PipeANN'], marker=markers['PipeANN'], linestyle='--', label='PipeANN QPS')
    ]

    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.98),
               ncol=3, frameon=False, columnspacing=1.5, labelspacing=0.5)

    # 留出顶部空间
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    
    save_path = os.path.expanduser('~/hpdic/AdaDisk/paper/TR2026/figures/cmp_pipeann.pdf')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path, bbox_inches='tight')
    print(f"PDF generated: {save_path}")

if __name__ == "__main__":
    draw_final_plots()