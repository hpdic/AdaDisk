#!/bin/bash

# 脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SCAN_LOG_DIR="$(dirname "$SCRIPT_DIR")/speed_scan_85_95"
mkdir -p "$SCAN_LOG_DIR"

echo "=== Starting Speed Optimization Scan (Target Recall: 85-95%) ==="

# ---------------------------------------------------------
# Group 1: SIFT -> 极低参数区
# 目标：把 Recall 降下来，看在这个区间的 QPS 爆发力
# ---------------------------------------------------------
DATASET_SIFT=("sift")
# R 需要很小
R_SIFT=(16 24 32)
# L 需要很小，现有数据 L=50 已经是 96% recall
L_SIFT=(20 30 40 50)
ALPHA_MIN="1.0"
# Sift 对 Alpha 敏感，细搜
ALPHA_MAXS_SIFT=("1.1" "1.2" "1.3" "1.5")

for DATASET in "${DATASET_SIFT[@]}"; do
    for R_VAL in "${R_SIFT[@]}"; do
        for L_VAL in "${L_SIFT[@]}"; do
            for MAX in "${ALPHA_MAXS_SIFT[@]}"; do
                
                LOG_FILE="$SCAN_LOG_DIR/${DATASET}_R${R_VAL}_L${L_VAL}_min${ALPHA_MIN}_max${MAX}.txt"
                if [ -f "$LOG_FILE" ]; then continue; fi

                echo "Running LOW-PARAM SIFT: R=$R_VAL | L=$L_VAL | Max=$MAX"
                
                export R="$R_VAL"
                export L="$L_VAL"
                export ALPHA_MIN="$ALPHA_MIN"
                export ALPHA_MAX="$MAX"
                export THREADS="32"

                bash "${SCRIPT_DIR}/run_exp_single.sh" "$DATASET" > "$LOG_FILE" 2>&1
            done
        done
    done
done

# ---------------------------------------------------------
# Group 2: Gist & Glove -> 中参数精细扫描
# 目标：在 L=50~150 之间找到 MCGI 比 Baseline 显著快的 Sweet Spot
# ---------------------------------------------------------
DATASETS_MID=("gist" "glove")
# R 保持中等，太小会导致连通性差，太大会拖慢速度
R_MID=(32 48 64)
# L 进行加密扫描，覆盖 85-95% 的关键区
L_MID=(60 80 100 120 140)
# Alpha 适当放宽
ALPHA_MAXS_MID=("1.2" "1.5" "1.8")

for DATASET in "${DATASETS_MID[@]}"; do
    for R_VAL in "${R_MID[@]}"; do
        for L_VAL in "${L_MID[@]}"; do
            for MAX in "${ALPHA_MAXS_MID[@]}"; do
                
                LOG_FILE="$SCAN_LOG_DIR/${DATASET}_R${R_VAL}_L${L_VAL}_min${ALPHA_MIN}_max${MAX}.txt"
                if [ -f "$LOG_FILE" ]; then continue; fi

                echo "Running MID-PARAM ${DATASET}: R=$R_VAL | L=$L_VAL | Max=$MAX"
                
                export R="$R_VAL"
                export L="$L_VAL"
                export ALPHA_MIN="$ALPHA_MIN"
                export ALPHA_MAX="$MAX"
                export THREADS="32"

                bash "${SCRIPT_DIR}/run_exp_single.sh" "$DATASET" > "$LOG_FILE" 2>&1
            done
        done
    done
done

echo "Speed Scan Done."