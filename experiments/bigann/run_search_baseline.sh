#!/bin/bash

# --- 1. 路径配置 ---
DISKANN_HOME="$HOME/hpdic/AdaDisk"
SEARCH_BIN="${DISKANN_HOME}/build/apps/search_disk_index"

# ✅ 修正 1: 索引路径指向 indices
# 注意：不要加 _disk.index 后缀
INDEX_PREFIX="$HOME/hpdic/sift1b_data/indices/diskann_base_R32_L50_B150G"

QUERY_FILE="$HOME/hpdic/sift1b_data/bigann_query.bin"  # 改为 .bin
GT_FILE="$HOME/hpdic/sift1b_data/bigann_gnd.bin"       # 改为 .bin

RESULT_OUTPUT="search_results.bin"

# --- 2. 搜索参数 ---
K=10                     # Top-10
L_LIST="10 20 40 80 100 120 140 160 180 200" # 不同的搜索队列长度
THREADS=128               # 线程数

# --- 3. 安全检查 ---
if [ ! -f "${INDEX_PREFIX}_disk.index" ]; then
    echo "❌ 找不到索引文件，请检查路径: ${INDEX_PREFIX}_disk.index"
    exit 1
fi

echo "🚀 开始测试旧索引: $INDEX_PREFIX"
echo "📂 Query: $QUERY_FILE"
echo "📂 GT: $GT_FILE"

"$SEARCH_BIN" \
  --data_type uint8 \
  --dist_fn l2 \
  --index_path_prefix "$INDEX_PREFIX" \
  --query_file "$QUERY_FILE" \
  --gt_file "$GT_FILE" \
  -K "$K" \
  -L $L_LIST \
  --result_path "$RESULT_OUTPUT" \
  --num_nodes_to_cache 10000 \
  -T "$THREADS"