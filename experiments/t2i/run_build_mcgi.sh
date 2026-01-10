#!/bin/bash
# SpaceV-1B Index Build Script (Baseline)

DATA_PATH="/home/cc/hpdic/deep1b_data/deep1b_base_1M.fbin"
INDEX_DIR="/home/cc/hpdic/deep1b_data/index_mcgi" #TODO Update this path as needed
INDEX_PREFIX="${INDEX_DIR}/deep1b"
BUILD_RAM_LIMIT=180  # 利用你的 251GB 内存，设为 200 可以极速构建

# 确保 build 目录存在
mkdir -p "${INDEX_DIR}"

echo "----------------------------------------------------------------"
echo "🚀 [Node0 修正版] 启动 DiskANN 构建 (MCGI)"
echo "----------------------------------------------------------------"
echo "📂 输入数据: $DATA_PATH"
echo "💾 输出索引: $INDEX_PREFIX"
echo "🧠 内存预算: ${BUILD_RAM_LIMIT} GB"
echo "----------------------------------------------------------------"

rm -f "${INDEX_PREFIX}"_disk.index
rm -f "${INDEX_PREFIX}"_mem*

start_time=$(date +%s)

echo "Starting Deep Index Build..."
~/hpdic/AdaDisk/build/apps/build_disk_index \
  --data_type float \
  --dist_fn l2 \
  --data_path "${DATA_PATH}" \
  --index_path_prefix "${INDEX_PREFIX}" \
  -R 32 \
  -L 50 \
  -B "${BUILD_RAM_LIMIT}" \
  -M "${BUILD_RAM_LIMIT}" \
  -T 96 \
  --use_amcgi \
  --alpha_min 0.5 \
  --alpha_max 1.2 \
  --lid_avg 16.5682 \
  --lid_std 5.9916   

end_time=$(date +%s)
duration=$((end_time - start_time))

echo "----------------------------------------------------------------"
echo "✅ 构建完成！耗时: $(($duration / 60)) 分钟"
echo "📊 索引文件位置: $INDEX_DIR"
echo "----------------------------------------------------------------"