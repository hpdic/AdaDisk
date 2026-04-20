#!/bin/bash

DATASET=${1:-gist}
THREADS=${THREADS:-32}

SCRIPT_DIR=$( cd $( dirname ${BASH_SOURCE[0]} ) &> /dev/null && pwd )
EXP_ROOT=$(dirname ${SCRIPT_DIR})

BUILDER=${EXP_ROOT}/../build/apps/build_disk_index
SEARCHER=${EXP_ROOT}/../build/apps/search_disk_index

DATA_DIR=${EXP_ROOT}/data/${DATASET}
BASE_BIN=${DATA_DIR}/${DATASET}_base.bin
LID_BIN=${DATA_DIR}/${DATASET}_lid.bin
QUERY_BIN=${DATA_DIR}/${DATASET}_query.bin
GT_BIN=${DATA_DIR}/${DATASET}_gt.bin

if [ ! -f ${BASE_BIN} ]; then
    echo 'Error: Base data missing at ' ${BASE_BIN}
    exit 1
fi

run_search() {
    PREFIX=$1
    NAME=$2
    OUT_DIR=$3
    IS_SHARED=$4
    
    echo '--- ' ${NAME} ' ---'
    printf '%-5s %-10s %-10s %-10s\n' 'L' 'QPS' 'Lat(us)' 'Recall'
    
    for SL in 50 100 150 200; do
        LOG=${OUT_DIR}/search_L${SL}.log
        
        if [ ${IS_SHARED} == '1' ] && [ -f ${LOG} ]; then
            LINE=$(grep -A 1 '========================' ${LOG} | tail -n 1)
        else
            ${SEARCHER} \
                --data_type float --dist_fn l2 \
                --index_path_prefix ${PREFIX} \
                --query_file ${QUERY_BIN} --gt_file ${GT_BIN} \
                -K 10 -L ${SL} --result_path ${OUT_DIR}/res --num_threads ${THREADS} > ${LOG} 2>&1
            
            LINE=$(grep -A 1 '========================' ${LOG} | tail -n 1)
        fi

        QPS=$(echo ${LINE} | awk '{print $3}')
        LAT=$(echo ${LINE} | awk '{print $4}')
        REC=$(echo ${LINE} | awk '{print $9}')
        
        if [ -z ${QPS} ]; then QPS='FAIL'; fi
        printf '%-5s %-10s %-10s %-10s\n' ${SL} ${QPS} ${LAT} ${REC}
    done
    echo ''
}

for CONFIG in '48 100 1.1 1.7' '32 100 1.0 2.0' '32 100 1.1 2.0'; do
    set -- ${CONFIG}
    R=$1
    L=$2
    ALPHA_MIN=$3
    ALPHA_MAX=$4
    
    SHARED_BASELINE_DIR=${EXP_ROOT}/results/${DATASET}_R${R}_L${L}_baseline_shared
    mkdir -p ${SHARED_BASELINE_DIR}

    CURRENT_MCGI_DIR=${EXP_ROOT}/results/${DATASET}_R${R}_L${L}_min${ALPHA_MIN}_max${ALPHA_MAX}
    mkdir -p ${CURRENT_MCGI_DIR}

    CURRENT_MCGI_LINEAR_DIR=${EXP_ROOT}/results/${DATASET}_R${R}_L${L}_min${ALPHA_MIN}_max${ALPHA_MAX}_linear
    mkdir -p ${CURRENT_MCGI_LINEAR_DIR}

    echo '=========================================================='
    echo 'Testing Config: R='${R}', L='${L}', Alpha=['${ALPHA_MIN}', '${ALPHA_MAX}']'
    echo '=========================================================='

    BASELINE_IDX_PREFIX=${SHARED_BASELINE_DIR}/idx
    BASELINE_IDX_FILE=${BASELINE_IDX_PREFIX}_disk.index

    if [ -f ${BASELINE_IDX_FILE} ]; then
        echo '=== [Cache Hit] Baseline index found. Skipping build. ==='
    else
        echo '=== [Cache Miss] Building Shared Baseline ==='
        ${BUILDER} \
            --data_type float --dist_fn l2 \
            --data_path ${BASE_BIN} \
            --index_path_prefix ${BASELINE_IDX_PREFIX} \
            -R ${R} -L ${L} -B 0.1 -M 1.0 -T ${THREADS} > ${SHARED_BASELINE_DIR}/build.log 2>&1
    fi

    MCGI_IDX_PREFIX=${CURRENT_MCGI_DIR}/idx
    echo '=== Building MCGI (Sigmoid) ==='
    ${BUILDER} \
        --data_type float --dist_fn l2 \
        --data_path ${BASE_BIN} \
        --index_path_prefix ${MCGI_IDX_PREFIX} \
        -R ${R} -L ${L} -B 0.1 -M 1.0 -T ${THREADS} \
        --use_mcgi --lid_path ${LID_BIN} \
        --alpha_min ${ALPHA_MIN} --alpha_max ${ALPHA_MAX} \
        --codebook_prefix ${BASELINE_IDX_PREFIX} \
        > ${CURRENT_MCGI_DIR}/build.log 2>&1

    MCGI_LINEAR_IDX_PREFIX=${CURRENT_MCGI_LINEAR_DIR}/idx
    echo '=== Building MCGI (Linear Ablation) ==='
    ${BUILDER} \
        --data_type float --dist_fn l2 \
        --data_path ${BASE_BIN} \
        --index_path_prefix ${MCGI_LINEAR_IDX_PREFIX} \
        -R ${R} -L ${L} -B 0.1 -M 1.0 -T ${THREADS} \
        --use_mcgi --use_linear 1 --lid_path ${LID_BIN} \
        --alpha_min ${ALPHA_MIN} --alpha_max ${ALPHA_MAX} \
        --codebook_prefix ${BASELINE_IDX_PREFIX} \
        > ${CURRENT_MCGI_LINEAR_DIR}/build.log 2>&1

    cp ${BASELINE_IDX_PREFIX}_pq_pivots.bin ${MCGI_IDX_PREFIX}_pq_pivots.bin
    cp ${BASELINE_IDX_PREFIX}_pq_compressed.bin ${MCGI_IDX_PREFIX}_pq_compressed.bin
    cp ${BASELINE_IDX_PREFIX}_pq_pivots.bin ${MCGI_LINEAR_IDX_PREFIX}_pq_pivots.bin
    cp ${BASELINE_IDX_PREFIX}_pq_compressed.bin ${MCGI_LINEAR_IDX_PREFIX}_pq_compressed.bin

    echo '=== Benchmarking ==='
    run_search ${BASELINE_IDX_PREFIX} 'Baseline' ${SHARED_BASELINE_DIR} 1
    run_search ${MCGI_IDX_PREFIX} 'MCGI (Sigmoid)' ${CURRENT_MCGI_DIR} 0
    run_search ${MCGI_LINEAR_IDX_PREFIX} 'MCGI (Linear)' ${CURRENT_MCGI_LINEAR_DIR} 0

    echo '=== Cleaning up large index files ==='
    rm -f ${CURRENT_MCGI_DIR}/*.index
    rm -f ${CURRENT_MCGI_DIR}/*.data
    rm -f ${CURRENT_MCGI_LINEAR_DIR}/*.index
    rm -f ${CURRENT_MCGI_LINEAR_DIR}/*.data

done

#
# Example output:
#
# (fluxvec) cc@uc-a100:~/hpdic/AdaDisk/experiments/scripts$ 
# (fluxvec) cc@uc-a100:~/hpdic/AdaDisk/experiments/scripts$ python find_best_params.py 
# Found the following configurations where MCGI significantly outperforms Baseline in Recall:

# File: gist_R48_min1.1_max1.7.txt
#   Search Depth L: 150
#   Baseline Recall: 93.97
#   MCGI Recall: 94.5
#   Improvement: +0.53%

# File: gist_R32_min1.0_max2.0.txt
#   Search Depth L: 150
#   Baseline Recall: 90.52
#   MCGI Recall: 91.03
#   Improvement: +0.51%

# File: gist_R32_min1.1_max2.0.txt
#   Search Depth L: 150
#   Baseline Recall: 90.52
#   MCGI Recall: 91.02
#   Improvement: +0.5%

# (fluxvec) cc@uc-a100:~/hpdic/AdaDisk/experiments/scripts$ ./run_ablation.sh 
# ==========================================================
# Testing Config: R=48, L=100, Alpha=[1.1, 1.7]
# ==========================================================
# === [Cache Miss] Building Shared Baseline ===
# === Building MCGI (Sigmoid) ===
# === Building MCGI (Linear Ablation) ===
# === Benchmarking ===
# ---  Baseline  ---
# L     QPS        Lat(us)    Recall    
# 50    144.52     221064.72  79.52     
# 100   677.75     46612.40   90.46     
# 150   453.32     69746.13   94.33     
# 200   339.11     93254.41   96.42     

# ---  MCGI (Sigmoid)  ---
# L     QPS        Lat(us)    Recall    
# 50    1180.66    26718.00   79.72     
# 100   642.50     49113.42   90.62     
# 150   432.94     73007.65   94.41     
# 200   331.21     95422.35   96.30     

# ---  MCGI (Linear)  ---
# L     QPS        Lat(us)    Recall    
# 50    212.74     146145.38  79.71     
# 100   228.81     137291.52  90.57     
# 150   197.39     159959.95  94.58     
# 200   175.89     179421.34  96.45     

# === Cleaning up large index files ===
# ==========================================================
# Testing Config: R=32, L=100, Alpha=[1.0, 2.0]
# ==========================================================
# === [Cache Miss] Building Shared Baseline ===
# === Building MCGI (Sigmoid) ===
# === Building MCGI (Linear Ablation) ===
# === Benchmarking ===
# ---  Baseline  ---
# L     QPS        Lat(us)    Recall    
# 50    931.92     34020.29   74.92     
# 100   654.69     48292.50   85.87     
# 150   442.36     71508.75   90.62     
# 200   333.41     94921.88   93.04     

# ---  MCGI (Sigmoid)  ---
# L     QPS        Lat(us)    Recall    
# 50    1139.95    27704.34   74.51     
# 100   628.74     50302.45   85.78     
# 150   432.28     73156.02   90.74     
# 200   327.62     96552.63   93.46     

# ---  MCGI (Linear)  ---
# L     QPS        Lat(us)    Recall    
# 50    285.34     110247.90  74.68     
# 100   225.19     139769.58  86.04     
# 150   181.37     173657.18  91.23     
# 200   158.29     199304.33  93.52     

# === Cleaning up large index files ===
# ==========================================================
# Testing Config: R=32, L=100, Alpha=[1.1, 2.0]
# ==========================================================
# === [Cache Hit] Baseline index found. Skipping build. ===
# === Building MCGI (Sigmoid) ===
# === Building MCGI (Linear Ablation) ===
# === Benchmarking ===
# ---  Baseline  ---
# L     QPS        Lat(us)    Recall    
# 50    931.92     34020.29   74.92     
# 100   654.69     48292.50   85.87     
# 150   442.36     71508.75   90.62     
# 200   333.41     94921.88   93.04     

# ---  MCGI (Sigmoid)  ---
# L     QPS        Lat(us)    Recall    
# 50    238.58     133751.55  74.80     
# 100   608.48     52035.94   85.98     
# 150   419.51     75415.61   91.37     
# 200   322.87     97971.87   93.58     

# ---  MCGI (Linear)  ---
# L     QPS        Lat(us)    Recall    
# 50    287.88     109234.74  74.78     
# 100   219.75     142974.66  86.01     
# 150   177.19     177755.73  91.18     
# 200   145.25     216642.36  93.46     

# === Cleaning up large index files ===
# (fluxvec) cc@uc-a100:~/hpdic/AdaDisk/experiments/scripts$ 