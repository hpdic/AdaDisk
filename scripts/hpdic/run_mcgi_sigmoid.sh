#!/bin/bash

# ==========================================
# 1. 基础配置 (Basic Configuration)
# ==========================================

# DiskANN 安装路径
DISKANN_HOME="/home/cc/AdaDisk"
BUILDER_BIN="${DISKANN_HOME}/build/apps/build_disk_index"

# 数据目录
DATA_DIR="${DISKANN_HOME}/hpdic_data"

# 输入数据 (向量)
RAW_DATA="${DATA_DIR}/ingest_raw.bin"

# [关键] MCGI 需要的 LID 数据文件
# 注意：你需要确保这个文件存在！它的格式应该是 DiskANN 的二进制 float 数组
LID_DATA="${DATA_DIR}/ingest_lid.bin" 

# 输出索引前缀 (改个名字，避免覆盖原版索引)
INDEX_PREFIX="${DATA_DIR}/ingest_index_mcgi"

# ==========================================
# 2. MCGI 参数配置 (Hyper-parameters)
# ==========================================

# 动态 Alpha 的范围
# 逻辑：简单点用 MIN，难点用 MAX
MCGI_ALPHA_MIN="1.0"  
MCGI_ALPHA_MAX="1.5"

# DiskANN 构建参数
R=32
L=50
B=0.1   # RAM Budget (GB)
M=0.1   # Build RAM Budget (GB)
THREADS=8

# ==========================================
# 3. 环境检查
# ==========================================

if [ ! -f "$BUILDER_BIN" ]; then
    echo "❌ 错误: 找不到构建工具: $BUILDER_BIN"
    exit 1
fi

if [ ! -f "$RAW_DATA" ]; then
    echo "❌ 错误: 找不到原始数据: $RAW_DATA"
    echo "   请先运行 gen_data.py 生成数据。"
    exit 1
fi

# [MCGI 特有检查]
if [ ! -f "$LID_DATA" ]; then
    echo "⚠️  警告: 未检测到 LID 数据文件: $LID_DATA"
    echo "   如果没有这个文件，MCGI 模块会报错或退化回默认模式。"
    echo "   (请确保你有一个计算好 LID 并存为 .bin 格式的文件)"
    # 这里不强制退出，方便你测试报错逻辑，或者你之后会生成它
fi

# ==========================================
# 4. 启动构建 (Run)
# ==========================================

echo "----------------------------------------------------------------"
echo "🚀 启动 MCGI-DiskANN (Sigmoid Mode)"
echo "----------------------------------------------------------------"
echo "📂 向量数据: $RAW_DATA"
echo "📊 LID 数据: $LID_DATA"
echo "🎚️  Alpha范围: [$MCGI_ALPHA_MIN, $MCGI_ALPHA_MAX]"
echo "----------------------------------------------------------------"

# 调用 C++ 程序，传入新加的三个参数
"$BUILDER_BIN" \
    --data_type float \
    --dist_fn l2 \
    --data_path "$RAW_DATA" \
    --index_path_prefix "$INDEX_PREFIX" \
    -R "$R" \
    -L "$L" \
    -B "$B" \
    -M "$M" \
    -T "$THREADS" \
    --lid_path "$LID_DATA" \
    --alpha_min "$MCGI_ALPHA_MIN" \
    --alpha_max "$MCGI_ALPHA_MAX" \
    --use_mcgi

# ==========================================
# 5. 结果检查
# ==========================================

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ MCGI 索引构建成功！"
    echo "你可以检查日志中是否有 '[MCGI]' 开头的输出来确认 Sigmoid 逻辑是否生效。"
else
    echo ""
    echo "❌ 构建失败 (Exit Code: $EXIT_CODE)"
fi