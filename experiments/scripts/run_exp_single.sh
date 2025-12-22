#!/bin/bash

# 获取数据集名称 (默认 sift)
DATASET="${1:-sift}"

# ==========================================
# 1. 动态参数配置 (优先读取环境变量)
# ==========================================
R="${R:-32}"
L="${L:-50}"
ALPHA_MIN="${ALPHA_MIN:-1.0}"
ALPHA_MAX="${ALPHA_MAX:-1.2}"
THREADS="${THREADS:-32}"

# ==========================================
# 2. 路径自动计算 (完全复刻 run_exp.sh)
# ==========================================
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
EXP_ROOT="$(dirname "$SCRIPT_DIR")"

# 使用 experiments/bin 下的 HPDIC MOD 程序
BUILDER="${EXP_ROOT}/bin/build_disk_index"
SEARCHER="${EXP_ROOT}/bin/search_disk_index"

# 数据路径
DATA_DIR="${EXP_ROOT}/data/${DATASET}"
BASE_BIN="${DATA_DIR}/${DATASET}_base.bin"
LID_BIN="${DATA_DIR}/${DATASET}_lid.bin"
QUERY_BIN="${DATA_DIR}/${DATASET}_query.bin"
GT_BIN="${DATA_DIR}/${DATASET}_gt.bin"

# ==========================================
# 3. 定义独立输出目录 (Grid Search 专用)
# ==========================================
# 结果路径包含 Alpha 值，防止覆盖
RES_DIR="${EXP_ROOT}/results/${DATASET}_R${R}_L${L}_min${ALPHA_MIN}_max${ALPHA_MAX}"
mkdir -p "$RES_DIR/baseline"
mkdir -p "$RES_DIR/mcgi"

echo "=========================================================="
echo "Grid Search Task: ${DATASET}"
echo "Params: R=$R, L=$L, Alpha=[$ALPHA_MIN, $ALPHA_MAX]"
echo "Output Dir: ${RES_DIR}"
echo "=========================================================="

# 检查数据是否存在
if [ ! -f "$BASE_BIN" ]; then
    echo "Error: Base data missing at $BASE_BIN"
    exit 1
fi

# ==========================================
# 4. 运行 Baseline (如果有公用 Baseline 可跳过，这里为稳妥起见每次检查)
# ==========================================
BASELINE_IDX="$RES_DIR/baseline/idx_disk.index"

if [ -f "$BASELINE_IDX" ]; then
    echo "Baseline index exists. Skipping build."
else
    echo "=== Building Baseline ==="
    "$BUILDER" \
        --data_type float --dist_fn l2 \
        --data_path "$BASE_BIN" \
        --index_path_prefix "$RES_DIR/baseline/idx" \
        -R $R -L $L -B 0.1 -M 1.0 -T 16 > "$RES_DIR/baseline/build.log" 2>&1

    if [ $? -ne 0 ]; then
        echo "Error: Baseline Build Failed."
        exit 1
    fi
fi

# ==========================================
# 5. 运行 MCGI
# ==========================================
MCGI_IDX="$RES_DIR/mcgi/idx_disk.index"
echo "=== Building MCGI ==="

"$BUILDER" \
    --data_type float --dist_fn l2 \
    --data_path "$BASE_BIN" \
    --index_path_prefix "$RES_DIR/mcgi/idx" \
    -R $R -L $L -B 0.1 -M 1.0 -T 16 \
    --use_mcgi --lid_path "$LID_BIN" \
    --alpha_min $ALPHA_MIN --alpha_max $ALPHA_MAX \
    --codebook_prefix "$RES_DIR/baseline/idx" \
    > "$RES_DIR/mcgi/build.log" 2>&1

if [ $? -ne 0 ]; then
    echo "Error: MCGI Build Failed."
    tail -n 5 "$RES_DIR/mcgi/build.log"
    exit 1
fi

# ==========================================
# 6. 同步 PQ 文件
# ==========================================
cp "$RES_DIR/baseline/idx_pq_pivots.bin" "$RES_DIR/mcgi/idx_pq_pivots.bin"
cp "$RES_DIR/baseline/idx_pq_compressed.bin" "$RES_DIR/mcgi/idx_pq_compressed.bin"

# ==========================================
# 7. 搜索对比 (输出 QPS)
# ==========================================
echo "=== Benchmarking ==="

# 这里的逻辑稍微简化，直接输出到标准输出，由 full_scan.sh 捕获
run_search() {
    PREFIX=$1
    NAME=$2
    OUT_DIR=$3
    echo "--- $NAME ---"
    printf "%-5s %-10s %-10s %-10s\n" "L" "QPS" "Lat(us)" "Recall"
    for SL in 50 100 150 200; do
        LOG="$OUT_DIR/search_L${SL}.log"
        "$SEARCHER" \
            --data_type float --dist_fn l2 \
            --index_path_prefix "$PREFIX" \
            --query_file "$QUERY_BIN" --gt_file "$GT_BIN" \
            -K 10 -L $SL --result_path "$OUT_DIR/res" --num_threads 32 > "$LOG" 2>&1
        
        # 提取结果
        LINE=$(grep -A 1 "========================" "$LOG" | tail -n 1)
        QPS=$(echo $LINE | awk '{print $3}')
        LAT=$(echo $LINE | awk '{print $4}')
        REC=$(echo $LINE | awk '{print $9}')
        
        if [ -z "$QPS" ]; then QPS="FAIL"; fi
        printf "%-5s %-10s %-10s %-10s\n" "$SL" "$QPS" "$LAT" "$REC"
    done
    echo ""
}

run_search "$RES_DIR/baseline/idx" "Baseline" "$RES_DIR/baseline"
run_search "$RES_DIR/mcgi/idx" "MCGI" "$RES_DIR/mcgi"

echo "=== Experiment Complete ==="