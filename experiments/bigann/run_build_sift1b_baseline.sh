#!/bin/bash
set -e  # 遇到错误立即停止

# --- 1. 路径配置 ---
# 你的 AdaDisk 源码目录 (位于 ~/hpdic/AdaDisk)
DISKANN_HOME="$HOME/hpdic/AdaDisk"

# 编译好的程序路径 (通常在 build/apps 下)
BUILDER_BIN="${DISKANN_HOME}/build/apps/build_disk_index"

# 数据绝对路径
RAW_DATA="$HOME/hpdic/sift1b_data/sift1b_base.bin"

# 输出路径 (放在 NVMe 上速度最快)
OUTPUT_DIR="$HOME/hpdic/sift1b_data/indices"

# --- 2. 核心参数 (R32/L50 性价比最高) ---
R_VAL=32           
L_VAL=50           
PQ_BYTES=16        

# --- 3. 内存与性能优化 ---
RAM_BUDGET=200      # the largest possible patch size is 199 GB
THREADS=128         # check htop

# --- 4. 自动命名 ---
INDEX_NAME="diskann_base_R${R_VAL}_L${L_VAL}_B${RAM_BUDGET}G"
INDEX_PREFIX="${OUTPUT_DIR}/${INDEX_NAME}"

# --- 5. 安全检查 (防止路径写错) ---
if [ ! -f "$RAW_DATA" ]; then
    echo "❌ 找不到数据文件: $RAW_DATA"
    exit 1
fi

if [ ! -f "$BUILDER_BIN" ]; then
    echo "❌ 找不到构建程序: $BUILDER_BIN"
    echo "   请检查编译目录是否正确，或者是否编译成功"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# --- 6. 执行 ---
echo "----------------------------------------------------------------"
echo "🚀 [Node0 修正版] 启动 DiskANN 构建"
echo "----------------------------------------------------------------"
echo "📂 程序位置: $BUILDER_BIN"
echo "📂 输入数据: $RAW_DATA"
echo "💾 输出索引: $INDEX_PREFIX"
echo "⚙️  算法参数: R=${R_VAL}, L=${L_VAL}"
echo "🧠 内存预算: ${RAM_BUDGET} GB"
echo "🧵 线程数量: $THREADS"
echo "----------------------------------------------------------------"

# 清理旧文件防止冲突
rm -f "${INDEX_PREFIX}"*

start_time=$(date +%s)

"$BUILDER_BIN" \
    --data_type uint8 \
    --dist_fn l2 \
    --data_path "$RAW_DATA" \
    --index_path_prefix "$INDEX_PREFIX" \
    -R "$R_VAL" \
    -L "$L_VAL" \
    -B "$RAM_BUDGET" \
    -M "$RAM_BUDGET" \
    -T "$THREADS" \
    --build_PQ_bytes "$PQ_BYTES"

end_time=$(date +%s)
duration=$((end_time - start_time))

echo "----------------------------------------------------------------"
echo "✅ 构建完成！耗时: $(($duration / 60)) 分钟"
echo "📊 索引文件位置: $OUTPUT_DIR"
echo "----------------------------------------------------------------"