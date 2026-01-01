#!/bin/bash
set -e  # 遇到任何错误立刻停止，防止连锁反应

# ==========================================
# SIFT1B Index Construction (Refactored & Safe)
# ==========================================

# --- 1. 核心配置 (在这里修改，不用动下面) ---
DISKANN_HOME="/home/cc/AdaDisk"
BUILDER_BIN="${DISKANN_HOME}/build/apps/build_disk_index"
RAW_DATA="/dev/shm/sift1b_base.bin"
OUTPUT_DIR="/home/cc/sift1b_data/indices"

# 针对硬盘空间受限 + Paper Reasonable 的参数
R_VAL=20           # 图的度数 (学术界常用区间底线，显著省空间)
L_VAL=40           # 构建列表大小 (R * 2，保证构建质量)
PQ_BYTES=16        # 压缩向量大小 (关键！防止中间文件膨胀)

# 内存控制
BUILD_RAM_GB=100   # (-M) 构建过程最大内存，你有256G，给100G很安全
SEARCH_RAM_GB=100  # (-B) 搜索/缓存预算，给足防止频繁IO
THREADS=64         # 线程数

# --- 2. 动态生成路径 (解决手动改名的痛苦) ---
# 文件名会自动变成: diskann_base_R20_L40
INDEX_NAME="diskann_base_R${R_VAL}_L${L_VAL}"
INDEX_PREFIX="${OUTPUT_DIR}/${INDEX_NAME}"

# --- 3. 安全检查 ---
if [ ! -f "$RAW_DATA" ]; then
    echo "❌ 致命错误: 找不到源数据文件: $RAW_DATA"
    exit 1
fi

if [ ! -f "$BUILDER_BIN" ]; then
    echo "❌ 致命错误: 找不到构建程序: $BUILDER_BIN"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# --- 4. 自动清理战场 ---
# ⚠️ 注意: 这里只清理 "当前参数" 对应的旧文件。
# 如果你之前跑过 R32 的废文件，请手动删一次，或者取消下面注释行的注释
# rm -f "${OUTPUT_DIR}/diskann_base_R32_L50*" 

echo "----------------------------------------------------------------"
echo "🚀 启动 DiskANN 构建任务"
echo "----------------------------------------------------------------"
echo "📂 源数据:  $RAW_DATA"
echo "💾 输出前缀: $INDEX_PREFIX"
echo "🔧 参数配置: R=${R_VAL}, L=${L_VAL}, PQ=${PQ_BYTES} bytes"
echo "🧠 内存限制: Build=${BUILD_RAM_GB}G, Limit=${SEARCH_RAM_GB}G"
echo "🧵 线程数:  $THREADS"
echo "----------------------------------------------------------------"

# 清理当前配置可能残留的旧文件，确保从零开始
echo "🧹 清理同名旧文件 (如果存在)..."
rm -f "${INDEX_PREFIX}"*

# --- 5. 执行构建 ---
# start_time=$(date +%s)

"$BUILDER_BIN" \
    --data_type uint8 \
    --dist_fn l2 \
    --data_path "$RAW_DATA" \
    --index_path_prefix "$INDEX_PREFIX" \
    -R "$R_VAL" \
    -L "$L_VAL" \
    -B "$SEARCH_RAM_GB" \
    -M "$BUILD_RAM_GB" \
    -T "$THREADS" \
    --build_PQ_bytes "$PQ_BYTES"

# --- 6. 结果反馈 ---
if [ $? -eq 0 ]; then
    echo "----------------------------------------------------------------"
    echo "✅ 成功！构建完成。"
    echo "📊 生成文件列表："
    ls -lh "${INDEX_PREFIX}"*
    echo "----------------------------------------------------------------"
else
    echo "❌ 失败！构建程序非正常退出。"
    exit 1
fi