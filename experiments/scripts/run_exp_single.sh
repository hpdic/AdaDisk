#!/bin/bash

# 获取数据集名称 (默认 sift)
DATASET="${1:-sift}"

# ==========================================
# 1. 动态参数配置
# ==========================================
R="${R:-32}"
L="${L:-50}"
ALPHA_MIN="${ALPHA_MIN:-1.0}"
ALPHA_MAX="${ALPHA_MAX:-1.2}"
THREADS="${THREADS:-32}"

# ==========================================
# 2. 路径配置
# ==========================================
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
EXP_ROOT="$(dirname "$SCRIPT_DIR")"

# 指向 build/apps 下的程序
BUILDER="${EXP_ROOT}/../build/apps/build_disk_index"
SEARCHER="${EXP_ROOT}/../build/apps/search_disk_index"

# 数据路径
DATA_DIR="${EXP_ROOT}/data/${DATASET}"
BASE_BIN="${DATA_DIR}/${DATASET}_base.bin"
LID_BIN="${DATA_DIR}/${DATASET}_lid.bin"
QUERY_BIN="${DATA_DIR}/${DATASET}_query.bin"
GT_BIN="${DATA_DIR}/${DATASET}_gt.bin"

# ==========================================
# 3. 定义目录结构 (关键修改)
# ==========================================

# [公共目录] 存放 Baseline (只跟 R, L 有关，跟 Alpha 无关)
# 这样所有 Alpha 不同的实验都会复用这个 Baseline
SHARED_BASELINE_DIR="${EXP_ROOT}/results/${DATASET}_R${R}_L${L}_baseline_shared"
mkdir -p "$SHARED_BASELINE_DIR"

# [当前实验目录] 存放 MCGI (跟 Alpha 有关)
CURRENT_MCGI_DIR="${EXP_ROOT}/results/${DATASET}_R${R}_L${L}_min${ALPHA_MIN}_max${ALPHA_MAX}"
mkdir -p "$CURRENT_MCGI_DIR"

echo "=========================================================="
echo "Grid Search Task: ${DATASET}"
echo "Params: R=$R, L=$L, Alpha=[$ALPHA_MIN, $ALPHA_MAX]"
echo "Shared Baseline: $SHARED_BASELINE_DIR"
echo "Current MCGI:    $CURRENT_MCGI_DIR"
echo "=========================================================="

if [ ! -f "$BASE_BIN" ]; then
    echo "Error: Base data missing at $BASE_BIN"
    exit 1
fi

# ==========================================
# 4. 智能 Baseline 构建 (复用机制)
# ==========================================
BASELINE_IDX_PREFIX="$SHARED_BASELINE_DIR/idx"
BASELINE_IDX_FILE="${BASELINE_IDX_PREFIX}_disk.index"

if [ -f "$BASELINE_IDX_FILE" ]; then
    echo "=== [Cache Hit] Baseline index found. Skipping build. ==="
else
    echo "=== [Cache Miss] Building Shared Baseline ==="
    "$BUILDER" \
        --data_type float --dist_fn l2 \
        --data_path "$BASE_BIN" \
        --index_path_prefix "$BASELINE_IDX_PREFIX" \
        -R $R -L $L -B 0.1 -M 1.0 -T 16 > "$SHARED_BASELINE_DIR/build.log" 2>&1

    if [ $? -ne 0 ]; then
        echo "Error: Baseline Build Failed."
        exit 1
    fi
fi

# ==========================================
# 5. 构建 MCGI (使用 Baseline 的 PQ 码本)
# ==========================================
MCGI_IDX_PREFIX="$CURRENT_MCGI_DIR/idx"
MCGI_IDX_FILE="${MCGI_IDX_PREFIX}_disk.index"

echo "=== Building MCGI ==="
"$BUILDER" \
    --data_type float --dist_fn l2 \
    --data_path "$BASE_BIN" \
    --index_path_prefix "$MCGI_IDX_PREFIX" \
    -R $R -L $L -B 0.1 -M 1.0 -T 16 \
    --use_mcgi --lid_path "$LID_BIN" \
    --alpha_min $ALPHA_MIN --alpha_max $ALPHA_MAX \
    --codebook_prefix "$BASELINE_IDX_PREFIX" \
    > "$CURRENT_MCGI_DIR/build.log" 2>&1

if [ $? -ne 0 ]; then
    echo "Error: MCGI Build Failed."
    tail -n 10 "$CURRENT_MCGI_DIR/build.log"
    exit 1
fi

# ==========================================
# 6. 同步 PQ 文件 (搜索必须)
# ==========================================
# 直接使用 Baseline 的 PQ 文件，无需复制几十兆的数据，软链接即可，或者让 Searcher 指向它们
# 为简单起见，我们还是复制过去（PQ文件很小，几MB），防止 Searcher 报错
cp "${BASELINE_IDX_PREFIX}_pq_pivots.bin" "${MCGI_IDX_PREFIX}_pq_pivots.bin"
cp "${BASELINE_IDX_PREFIX}_pq_compressed.bin" "${MCGI_IDX_PREFIX}_pq_compressed.bin"

# ==========================================
# 7. 搜索对比 (智能缓存 + 清理)
# ==========================================
echo "=== Benchmarking ==="

# --- 定义搜索函数 ---
# 参数: 1=IndexPrefix, 2=Name, 3=OutputDir, 4=IsShared(0/1)
run_search() {
    PREFIX=$1
    NAME=$2
    OUT_DIR=$3
    IS_SHARED=$4
    
    echo "--- $NAME ---"
    printf "%-5s %-10s %-10s %-10s\n" "L" "QPS" "Lat(us)" "Recall"
    
    for SL in 50 100 150 200; do
        LOG="$OUT_DIR/search_L${SL}.log"
        
        # 逻辑：如果是共享Baseline且日志已存在，则直接读取结果，不跑搜索
        if [ "$IS_SHARED" == "1" ] && [ -f "$LOG" ]; then
            # 从旧日志提取结果
            LINE=$(grep -A 1 "========================" "$LOG" | tail -n 1)
        else
            # 运行搜索
            "$SEARCHER" \
                --data_type float --dist_fn l2 \
                --index_path_prefix "$PREFIX" \
                --query_file "$QUERY_BIN" --gt_file "$GT_BIN" \
                -K 10 -L $SL --result_path "$OUT_DIR/res" --num_threads 32 > "$LOG" 2>&1
            
            LINE=$(grep -A 1 "========================" "$LOG" | tail -n 1)
        fi

        # 解析输出
        QPS=$(echo $LINE | awk '{print $3}')
        LAT=$(echo $LINE | awk '{print $4}')
        REC=$(echo $LINE | awk '{print $9}')
        
        if [ -z "$QPS" ]; then QPS="FAIL"; fi
        printf "%-5s %-10s %-10s %-10s\n" "$SL" "$QPS" "$LAT" "$REC"
    done
    echo ""
}

# 1. 跑 Baseline (开启复用模式 IS_SHARED=1)
run_search "$BASELINE_IDX_PREFIX" "Baseline" "$SHARED_BASELINE_DIR" 1

# 2. 跑 MCGI (不复用，每次都是新的 IS_SHARED=0)
run_search "$MCGI_IDX_PREFIX" "MCGI" "$CURRENT_MCGI_DIR" 0

echo "=== Experiment Complete ==="

# ==========================================
# 8. 空间清理 (关键步骤)
# ==========================================
echo "=== Cleaning up large index files ==="
# 注意：千万别删 SHARED_BASELINE_DIR，因为下一个循环还要用！
# 只删除当前 MCGI 的 .index 文件 (通常几GB)
rm -f "$CURRENT_MCGI_DIR"/*.index
rm -f "$CURRENT_MCGI_DIR"/*.data # 如果有
# 保留 build.log 和 search_*.log 以备查验

echo "Cleanup done. Large binaries removed."