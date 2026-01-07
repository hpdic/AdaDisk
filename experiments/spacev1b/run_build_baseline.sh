#!/bin/bash
# SpaceV-1B Index Build Script (Baseline)

DATA_PATH="/home/cc/hpdic/spacev1b_data/spacev1b_base.bin"
INDEX_PREFIX="/home/cc/hpdic/spacev1b_data/spacev1b_index"
BUILD_RAM_LIMIT=200  # 利用你的 251GB 内存，设为 200 可以极速构建

# 确保 build 目录存在
mkdir -p $(dirname "${INDEX_PREFIX}")

echo "Starting SpaceV-1B Index Build..."
# 注意 data_type 是 int8
~/hpdic/AdaDisk/build/apps/build_disk_index \
  --data_type int8 \
  --dist_fn l2 \
  --data_path "${DATA_PATH}" \
  --index_path_prefix "${INDEX_PREFIX}" \
  -R 32 \
  -L 50 \
  -B "${BUILD_RAM_LIMIT}" \
  -M 32 \
  -T 96