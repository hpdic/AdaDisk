#!/bin/bash

# ==========================================
# SIFT1B Baseline Index Construction
# ==========================================

# 1. 基础路径配置
DISKANN_HOME="/home/cc/AdaDisk"
BUILDER_BIN="${DISKANN_HOME}/build/apps/build_disk_index"

# 2. SIFT1B 数据路径 (指向你刚才转换好的 bin)
DATA_DIR="/home/cc/sift1b_data"
RAW_DATA="${DATA_DIR}/sift1b_base.bin"

# 3. 输出路径
# 建议单独建个文件夹存索引
INDEX_DIR="${DATA_DIR}/indices"
mkdir -p "$INDEX_DIR"
INDEX_PREFIX="${INDEX_DIR}/diskann_base_R64_L100"

# ==========================================
# 4. 关键参数配置 (针对 Haswell 64GB 节点)
# ==========================================

R=64            # 图的度数 (Standard for Billion Scale)
L=100           # 构建时的候选列表大小
B=48            # build_DRAM_limit (GB): 给构建过程分配 48GB 内存 (留 16GB 给系统)
M=48            # build_memory_limit (GB): 同样的限制
THREADS=40      # 你的 CPU 线程数 (拉满)

# ==========================================
# 5. 执行构建
# ==========================================

if [ ! -f "$RAW_DATA" ]; then
    echo "❌ 错误: 找不到数据文件 $RAW_DATA"
    exit 1
fi

echo "----------------------------------------------------------------"
echo "🚀 开始构建 SIFT1B (1 Billion) 索引..."
echo "📂 输入数据: $RAW_DATA"
echo "💾 输出路径: $INDEX_PREFIX"
echo "⚙️  配置: R=$R, L=$L, RAM Limit=${B}GB, Threads=$THREADS"
echo "----------------------------------------------------------------"

# 记录开始时间
start_time=$(date +%s)

"$BUILDER_BIN" \
    --data_type float \
    --dist_fn l2 \
    --data_path "$RAW_DATA" \
    --index_path_prefix "$INDEX_PREFIX" \
    -R "$R" \
    -L "$L" \
    -B "$B" \
    -M "$M" \
    -T "$THREADS" 

# 计算耗时
end_time=$(date +%s)
duration=$((end_time - start_time))
hours=$((duration / 3600))
minutes=$(((duration % 3600) / 60))

if [ $? -eq 0 ]; then
    echo "----------------------------------------------------------------"
    echo "✅ SIFT1B 索引构建成功！"
    echo "⏱️  总耗时: ${hours} 小时 ${minutes} 分钟"
    echo "----------------------------------------------------------------"
    ls -lh "${INDEX_PREFIX}"*
else
    echo "❌ 构建失败！"
    exit 1
fi