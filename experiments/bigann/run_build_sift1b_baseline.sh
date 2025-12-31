#!/bin/bash

# ==========================================
# SIFT1B Baseline Index Construction (Final)
# ==========================================

# 1. 路径配置
DISKANN_HOME="/home/cc/AdaDisk"
BUILDER_BIN="${DISKANN_HOME}/build/apps/build_disk_index"
RAW_DATA="/home/cc/sift1b_data/sift1b_base.bin"
INDEX_PREFIX="/home/cc/sift1b_data/indices/diskann_base_R64_L100"

# 2. 关键参数 (针对 Haswell 64GB 节点)
R=64            # 图度数
L=100           # 构建列表大小
B=48            # 内存限制 48GB (机器有64G，留点余量)
M=48            # 同样设为 48GB
THREADS=40      # 线程拉满

# 3. 安全检查
if [ ! -f "$RAW_DATA" ]; then
    echo "❌ 错误: 找不到数据文件 $RAW_DATA"
    exit 1
fi

# 4. 自动清理旧文件 (防止上次失败的残留干扰)
rm -f "${INDEX_PREFIX}"*

echo "----------------------------------------------------------------"
echo "🚀 开始构建 SIFT1B 索引 (uint8 修正版)..."
echo "📂 输入: $RAW_DATA"
echo "💾 输出: $INDEX_PREFIX"
echo "⚙️  参数: R=$R, L=$L, RAM=$B GB, Threads=$THREADS"
echo "----------------------------------------------------------------"

# 5. 执行构建 (注意这里是 uint8)
"$BUILDER_BIN" \
    --data_type uint8 \
    --dist_fn l2 \
    --data_path "$RAW_DATA" \
    --index_path_prefix "$INDEX_PREFIX" \
    -R "$R" \
    -L "$L" \
    -B "$B" \
    -M "$M" \
    -T "$THREADS" 

if [ $? -eq 0 ]; then
    echo "✅ 构建成功！"
    ls -lh "${INDEX_PREFIX}"*
else
    echo "❌ 构建失败！"
    exit 1
fi