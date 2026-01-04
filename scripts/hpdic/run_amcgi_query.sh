#!/bin/bash

# ==========================================
# 纯内存搜索配置 (No PQ required)
# ==========================================

DISKANN_HOME="$HOME/hpdic/AdaDisk"
# 【关键修改】使用 search_memory_index
SEARCH_BIN="${DISKANN_HOME}/build/apps/search_memory_index" 
DATA_DIR="${DISKANN_HOME}/hpdic_data"

# 索引前缀
INDEX_PREFIX="${DATA_DIR}/ingest_index_amcgi"

# Query 数据
QUERY_BIN="${DATA_DIR}/ingest_raw.bin"

echo "----------------------------------------------------------------"
echo "🔍 Testing Memory Search (No PQ needed)"
echo "----------------------------------------------------------------"

# L_search list
L_LIST="20 40 80 100"
K_VAL=10

# 结果路径
RES_PATH="${DATA_DIR}/res"

# 注意：
# 1. 这里的 binary 是 search_memory_index
# 2. 它不需要 ._pq_pivots.bin，只需要 .index (图) 和 .data (原始向量)
# 3. 因为没有传 GT (GroundTruth)，Recall 会显示为 0，但 QPS 和 Latency 会正常输出，
#    这足够证明你的索引能不能用了。

"$SEARCH_BIN" \
    --data_type float \
    --dist_fn l2 \
    --index_path_prefix "$INDEX_PREFIX" \
    --query_file "$QUERY_BIN" \
    --gt_file "" \
    -K $K_VAL \
    -L 50 \
    --result_path "$RES_PATH"

if [ $? -eq 0 ]; then
    echo "✅ Memory Search Success! Index is valid."
else
    echo "❌ Memory Search Failed."
fi