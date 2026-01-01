#!/bin/bash

# ==========================================
# SIFT1B "Flash Run" Index Construction (SSD Optimized)
# ==========================================

# 1. 路径配置 (请确保这些路径都在 SSD 上!)
# 假设 SSD 挂载在 /mnt/ssd 或类似位置，请根据实际情况修改
DISKANN_HOME="/home/cc/AdaDisk"
BUILDER_BIN="${DISKANN_HOME}/build/apps/build_disk_index"

# ⚠️ 注意: 输入数据和输出索引必须都在 SSD 上才能发挥速度
# 如果 SSD 空间极度紧张，可以尝试把 RAW_DATA 放 HDD (如果读取速度能跟上)，但建议放 SSD
RAW_DATA="/home/cc/sift1b_data/sift1b_base.bin" 
INDEX_PREFIX="/home/cc/sift1b_data/indices/diskann_base_R32_L50"

# 2. 关键参数 (针对 Ice Lake 256GB RAM + 480GB SSD)
R=32            # ⬇️ 降级: 节省约 120GB 空间，确保不爆盘
L=50            # ⬇️ 降级: 加速构建，先跑通再说
B=200           # ⬆️ 升级: 机器有256G，给200G让它在内存里狂奔，减少写盘
M=64            # ⬆️ 调整: 中间图度数，设为 64 保证质量不至于太差
THREADS=64      # ⬆️ 调整: SATA SSD 写入瓶颈，64线程通常比160线程更稳
PQ_BYTES=16     # 🔥【关键修改】: 将分片向量压缩到16字节，防止硬盘爆炸！

# 3. 安全检查
if [ ! -f "$RAW_DATA" ]; then
    echo "❌ 错误: 找不到数据文件 $RAW_DATA"
    echo "请确认 SSD 上是否有数据，或者修改路径指向 HDD"
    exit 1
fi

# 4. 空间检查 (简单预估)
# 索引估计大小: 128G(数据) + ~128G(图 R=32) ≈ 256GB
# 加上输入数据 128GB，总共需要 ~384GB。480GB 硬盘应该剩 ~440GB 可用，够用。

echo "----------------------------------------------------------------"
echo "🚀 开始构建 SIFT1B 索引 (SSD Flash Mode)..."
echo "📂 输入: $RAW_DATA"
echo "💾 输出: $INDEX_PREFIX"
echo "⚙️  参数: R=$R, L=$L, RAM=$B GB, M=$M, Threads=$THREADS"
echo "⚠️  注意: 请监控磁盘剩余空间 (df -h)"
echo "----------------------------------------------------------------"

# 确保目录绝对存在
mkdir -p /home/cc/sift1b_data/indices

# 清理之前的残留 (如果有)
rm -f /home/cc/sift1b_data/indices/diskann_base_R32_L50*

# 5. 执行构建 (uint8)
"$BUILDER_BIN" \
    --data_type uint8 \
    --dist_fn l2 \
    --data_path "$RAW_DATA" \
    --index_path_prefix "$INDEX_PREFIX" \
    -R "$R" \
    -L "$L" \
    -B "$B" \
    -M "$M" \
    -T "$THREADS" \
    --build_PQ_bytes "$PQ_BYTES"

if [ $? -eq 0 ]; then
    echo "✅ 构建成功！"
    ls -lh "${INDEX_PREFIX}"*
else
    echo "❌ 构建失败！"
    exit 1
fi