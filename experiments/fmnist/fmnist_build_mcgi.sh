#!/bin/bash
set -e

DISKANN_HOME=${HOME}/hpdic/AdaDisk
BUILDER_BIN=${DISKANN_HOME}/build/apps/build_disk_index
RAW_DATA=${HOME}/hpdic/data_fmnist/fashion_base.bin
OUTPUT_DIR=${HOME}/hpdic/data_fmnist/indices_mcgi
LID_FILE=${HOME}/hpdic/data_fmnist/fashion_lid.bin

ALPHA_MIN='1.0'
ALPHA_MAX='1.5'

R_VAL=32
L_VAL=50

RAM_BUDGET=2
THREADS=96

INDEX_NAME=diskann_mcgi_R${R_VAL}_L${L_VAL}_B${RAM_BUDGET}G
INDEX_PREFIX=${OUTPUT_DIR}/${INDEX_NAME}

mkdir -p ${OUTPUT_DIR}
rm -f ${INDEX_PREFIX}*

BASE_PREFIX="${HOME}/hpdic/data_fmnist/indices/diskann_base_R32_L50_B2G"
cp ${BASE_PREFIX}_pq_pivots.bin ${INDEX_PREFIX}_pq_pivots.bin
cp ${BASE_PREFIX}_pq_compressed.bin ${INDEX_PREFIX}_pq_compressed.bin

echo 'Start building fmnist mcgi...'

${BUILDER_BIN} \
    --data_type float \
    --dist_fn l2 \
    --data_path ${RAW_DATA} \
    --index_path_prefix ${INDEX_PREFIX} \
    -R ${R_VAL} \
    -L ${L_VAL} \
    -B ${RAM_BUDGET} \
    -M ${RAM_BUDGET} \
    -T ${THREADS} \
    --use_amcgi \
    --alpha_min ${ALPHA_MIN} \
    --alpha_max ${ALPHA_MAX}