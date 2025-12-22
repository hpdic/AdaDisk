#!/bin/bash

# 数据集列表
DATASETS=("sift" "glove" "gist")

# R 值列表
R_VALUES=(32 48 64)

# Alpha Min/Max 列表 (包含 1.7 和 2.0)
ALPHA_MINS=("1.0" "1.1")
ALPHA_MAXS=("1.1" "1.2" "1.3" "1.4" "1.5" "1.6" "1.7" "2.0")

# 脚本所在目录 (自动获取)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# 确保日志目录在 experiments 下
SCAN_LOG_DIR="$(dirname "$SCRIPT_DIR")/fullscan"
mkdir -p "$SCAN_LOG_DIR"

echo "=========================================="
echo "Starting Massive Grid Search"
echo "Logs dir: $SCAN_LOG_DIR"
echo "=========================================="

for DATASET in "${DATASETS[@]}"; do
    for R_VAL in "${R_VALUES[@]}"; do
        for MIN in "${ALPHA_MINS[@]}"; do
            for MAX in "${ALPHA_MAXS[@]}"; do
                
                # 跳过 Min >= Max
                if (( $(echo "$MIN >= $MAX" | bc -l) )); then continue; fi

                LOG_FILE="$SCAN_LOG_DIR/${DATASET}_R${R_VAL}_min${MIN}_max${MAX}.txt"

                # 跳过已完成
                if [ -f "$LOG_FILE" ]; then
                    echo "[Skip] Exists: $LOG_FILE"
                    continue
                fi

                echo "Running: $DATASET | R=$R_VAL | Min=$MIN | Max=$MAX"
                
                # 环境变量传递
                export R="$R_VAL"
                export L="100"
                export ALPHA_MIN="$MIN"
                export ALPHA_MAX="$MAX"
                export THREADS="32"

                # 调用刚才修复的 run_exp_single.sh
                bash "${SCRIPT_DIR}/run_exp_single.sh" "$DATASET" > "$LOG_FILE" 2>&1

                sleep 1
            done
        done
    done
done
echo "All Done."