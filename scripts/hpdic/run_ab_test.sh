#!/bin/bash

# ================= 配置区 =================
DISKANN_HOME="${HOME}/hpdic/AdaDisk"
BUILDER_BIN="${DISKANN_HOME}/build/apps/build_disk_index"
SEARCH_BIN="${DISKANN_HOME}/build/apps/search_disk_index"

DATA_ROOT="${DISKANN_HOME}/hpdic_data"
RAW_DATA="${DATA_ROOT}/ingest_raw.bin"
LID_DATA="${DATA_ROOT}/ingest_lid.bin"
QUERY_DATA="${DATA_ROOT}/ingest_query.bin"
GT_DATA="${DATA_ROOT}/ingest_gt.bin"

# === 物理隔离目录 ===
DIR_BASE="${DATA_ROOT}/baseline"
DIR_MCGI="${DATA_ROOT}/mcgi"

PREFIX_BASE="${DIR_BASE}/disk_index"
PREFIX_MCGI="${DIR_MCGI}/disk_index"

# 参数
R=32
L=50
B=0.1
M=0.1
THREADS=8
ALPHA_MIN="1.0"
ALPHA_MAX="1.5"
SEARCH_L_LIST="10 20 40 80 100"
K_RECALL=10

# ================= 0. 环境初始化 =================
echo "============================================="
echo "🧹 [0/4] Cleaning & Creating Directories..."
echo "============================================="
rm -rf "$DIR_BASE"
rm -rf "$DIR_MCGI"
mkdir -p "$DIR_BASE"
mkdir -p "$DIR_MCGI"
echo "✅ Created separated folders."

# ================= 1. 构建 Baseline =================
echo "============================================="
echo "🏗️  [1/4] Building Baseline Index..."
echo "============================================="
"$BUILDER_BIN" \
    --data_type float --dist_fn l2 --data_path "$RAW_DATA" \
    --index_path_prefix "$PREFIX_BASE" \
    -R "$R" -L "$L" -B "$B" -M "$M" -T "$THREADS" > "${DIR_BASE}/build.log" 2>&1

if [ $? -eq 0 ]; then
    echo "✅ Baseline Build Done."
else
    echo "❌ Baseline Failed. Check log: ${DIR_BASE}/build.log"
    exit 1
fi

# ================= 2. 构建 MCGI (使用 Baseline 的 PQ) =================
echo "============================================="
echo "🏗️  [2/4] Building MCGI Index (Shared PQ)..."
echo "============================================="

# [关键修改] 加上 --codebook_prefix "$PREFIX_BASE"
# 这会让 MCGI 直接读取 Baseline 生成好的 PQ 文件，跳过容易出错的 PQ 训练步骤。
# 这样不仅能修复 Crash，还能保证构建出完整的 _disk.index 文件。

"$BUILDER_BIN" \
    --data_type float --dist_fn l2 --data_path "$RAW_DATA" \
    --index_path_prefix "$PREFIX_MCGI" \
    -R "$R" -L "$L" -B "$B" -M "$M" -T "$THREADS" \
    --lid_path "$LID_DATA" --alpha_min "$ALPHA_MIN" --alpha_max "$ALPHA_MAX" \
    --use_mcgi \
    --codebook_prefix "$PREFIX_BASE" > "${DIR_MCGI}/build.log" 2>&1

if [ $? -eq 0 ]; then
    echo "✅ MCGI Build Done."
else
    echo "❌ MCGI Failed. Check log: ${DIR_MCGI}/build.log"
    tail -n 10 "${DIR_MCGI}/build.log"
    exit 1
fi

# ================= 3. 同步 PQ 文件供 Search 使用 =================
echo "============================================="
echo "💉 [3/4] Syncing PQ files for Search..."
echo "============================================="
# 虽然构建时用了 Baseline 的 PQ，但 Search 程序默认会在 MCGI 目录下找同名前缀的文件
# 所以我们需要把 Baseline 的 PQ 文件拷一份到 mcgi 目录下，并重命名
cp "${PREFIX_BASE}_pq_pivots.bin" "${PREFIX_MCGI}_pq_pivots.bin"
cp "${PREFIX_BASE}_pq_compressed.bin" "${PREFIX_MCGI}_pq_compressed.bin"
echo "✅ PQ Files Ready."

# ================= 4. 搜索对比 (PK) =================
echo "============================================="
echo "⚔️  [4/4] Running Search Benchmark..."
echo "============================================="

run_benchmark() {
    WORK_DIR=$1
    PREFIX=$2
    NAME=$3
    
    echo "--- Benchmarking: $NAME ---"
    printf "%-5s %-10s %-15s %-10s\n" "L" "QPS" "Latency(us)" "Recall@${K_RECALL}"
    echo "------------------------------------------------"

    for SEARCH_L in $SEARCH_L_LIST; do
        LOG_FILE="${WORK_DIR}/search_L${SEARCH_L}.log"
        
        "$SEARCH_BIN" \
            --data_type float --dist_fn l2 \
            --index_path_prefix "$PREFIX" \
            --query_file "$QUERY_DATA" \
            --gt_file "$GT_DATA" \
            -K "$K_RECALL" \
            -L "$SEARCH_L" \
            --result_path "${WORK_DIR}/res" \
            --num_threads 1 > "$LOG_FILE" 2>&1

        DATA_LINE=$(grep -A 1 "========================" "$LOG_FILE" | tail -n 1)
        QPS=$(echo "$DATA_LINE" | awk '{print $3}')
        LATENCY=$(echo "$DATA_LINE" | awk '{print $4}')
        RECALL=$(echo "$DATA_LINE" | awk '{print $9}')

        if [ -z "$QPS" ]; then QPS="?"; fi
        printf "%-5s %-10s %-15s %-10s\n" "$SEARCH_L" "$QPS" "$LATENCY" "$RECALL"
    done
    echo ""
}

run_benchmark "$DIR_BASE" "$PREFIX_BASE" "Baseline"
run_benchmark "$DIR_MCGI" "$PREFIX_MCGI" "MCGI (Sigmoid)"

echo "============================================="
echo "✅ All Tests Finished."
echo "============================================="