#!/bin/bash

# ================= 动态配置区 =================
# 用法: bash run_exp.sh [sift | gist | glove]
DATASET="${1:-sift}" 

if [ "$DATASET" == "sift" ]; then
    R=32
    L=50
    ALPHA_MIN="1.0"
    ALPHA_MAX="1.5"
elif [ "$DATASET" == "gist" ]; then
    R=64    
    L=100
    ALPHA_MIN="1.0"
    ALPHA_MAX="1.5"
elif [ "$DATASET" == "glove" ]; then
    # === GloVe 配置 ===
    R=32
    L=50
    ALPHA_MIN="1.0"
    ALPHA_MAX="1.5"
else
    echo "Error: Unknown dataset: $DATASET"
    exit 1
fi

# ================= 路径自动计算 =================
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
EXP_ROOT="$(dirname "$SCRIPT_DIR")"

# 使用 experiments/bin 下的程序
BUILDER="${EXP_ROOT}/bin/build_disk_index"
SEARCHER="${EXP_ROOT}/bin/search_disk_index"

# 数据路径
DATA_DIR="${EXP_ROOT}/data/${DATASET}"
BASE_BIN="${DATA_DIR}/${DATASET}_base.bin"
LID_BIN="${DATA_DIR}/${DATASET}_lid.bin"
QUERY_BIN="${DATA_DIR}/${DATASET}_query.bin"
GT_BIN="${DATA_DIR}/${DATASET}_gt.bin"

# 结果路径
RES_DIR="${EXP_ROOT}/results/${DATASET}_R${R}_L${L}"
mkdir -p "$RES_DIR/baseline"
mkdir -p "$RES_DIR/mcgi"

echo "=========================================================="
echo "Running Experiment on ${DATASET}"
echo "Output Dir: ${RES_DIR}"
echo "=========================================================="

# 1. 检查数据
if [ ! -f "$BASE_BIN" ]; then
    echo "Error: Base data missing at $BASE_BIN"
    echo "Please run get_data.py or get_glove.py first."
    exit 1
fi

# 自动计算 LID
if [ ! -f "$LID_BIN" ]; then
    echo "Warning: LID missing. Computing automatically..."
    OPENBLAS_NUM_THREADS=1 python3 "${SCRIPT_DIR}/calc_lid.py" "$BASE_BIN"
    if [ ! -f "$LID_BIN" ]; then echo "Error: LID calculation failed."; exit 1; fi
fi

# 2. 运行 Baseline
echo "=== Building Baseline ==="
"$BUILDER" \
    --data_type float --dist_fn l2 \
    --data_path "$BASE_BIN" \
    --index_path_prefix "$RES_DIR/baseline/idx" \
    -R $R -L $L -B 0.1 -M 1.0 -T 16 > "$RES_DIR/baseline/build.log" 2>&1

if [ $? -ne 0 ]; then
    echo "Error: Baseline Build Failed. Check log:"
    tail -n 5 "$RES_DIR/baseline/build.log"
    exit 1
fi

# 3. 运行 MCGI
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
    echo "Error: MCGI Build Failed. Check log:"
    tail -n 5 "$RES_DIR/mcgi/build.log"
    exit 1
fi

# 4. 同步 PQ 文件 (Search 需要)
echo "=== Syncing PQ Files ==="
cp "$RES_DIR/baseline/idx_pq_pivots.bin" "$RES_DIR/mcgi/idx_pq_pivots.bin"
cp "$RES_DIR/baseline/idx_pq_compressed.bin" "$RES_DIR/mcgi/idx_pq_compressed.bin"

# 5. 搜索对比
echo "=== Benchmarking ==="

run_search() {
    PREFIX=$1
    NAME=$2
    OUT_DIR=$3
    echo "--- $NAME ---"
    printf "%-5s %-10s %-10s %-10s\n" "L" "QPS" "Lat(us)" "Recall"
    for SL in 10 20 40 80 100; do
        LOG="$OUT_DIR/search_L${SL}.log"
        "$SEARCHER" \
            --data_type float --dist_fn l2 \
            --index_path_prefix "$PREFIX" \
            --query_file "$QUERY_BIN" --gt_file "$GT_BIN" \
            -K 10 -L $SL --result_path "$OUT_DIR/res" --num_threads 1 > "$LOG" 2>&1
        
        # 提取结果
        LINE=$(grep -A 1 "========================" "$LOG" | tail -n 1)
        QPS=$(echo $LINE | awk '{print $3}')
        LAT=$(echo $LINE | awk '{print $4}')
        REC=$(echo $LINE | awk '{print $9}')
        
        # 防止 grep 为空
        if [ -z "$QPS" ]; then QPS="?"; fi
        
        printf "%-5s %-10s %-10s %-10s\n" "$SL" "$QPS" "$LAT" "$REC"
    done
    echo ""
}

run_search "$RES_DIR/baseline/idx" "Baseline" "$RES_DIR/baseline"
run_search "$RES_DIR/mcgi/idx" "MCGI" "$RES_DIR/mcgi"

echo "=== Experiment Complete ==="