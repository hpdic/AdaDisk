# After installing PipeANN with the C++ interface, on the root directory of PipeANN, run the following commands to prepare the data for GIST1M experiments. Make sure to adjust the paths according to your setup.

# 转换 base 文件
build/tests/utils/vecs_to_bin float /home/cc/hpdic/AdaDisk/experiments/data/gist/gist_base.fvecs data_gist.bin

# 转换 query 文件
build/tests/utils/vecs_to_bin float /home/cc/hpdic/AdaDisk/experiments/data/gist/gist_query.fvecs query_gist.bin

# 转换 groundtruth 文件（注意 GT 通常是 int 类型）
build/tests/utils/vecs_to_bin int32 /home/cc/hpdic/AdaDisk/experiments/data/gist/gist_groundtruth.ivecs gt_gist.bin

#
# R = 32
#

# build_disk_index <type> <data> <prefix> <R> <L> <PQ_bytes> <M_GB> <threads> <metric> <nbr_type>
build/tests/build_disk_index float data_gist.bin pipeann_gist_idx 32 150 32 64 32 l2 pq

# search_disk_index <type> <prefix> <threads> <beam_width> <query> <gt> <topk> <metric> <nbr_type> <mode> <mem_L> <Ls...>
build/tests/search_disk_index float pipeann_gist_idx 1 32 query_gist.bin gt_gist.bin 10 l2 pq 2 0 50 100 150 200

#
# Example output:
#
#      L   I/O Width         QPS  AvgLat(us)     P99 Lat   Mean Hops    Mean IOs   Recall@10
# =========================================================================================
#     50          32      279.37     3557.16    10949.00        0.00       83.04       56.42
#    100          32      202.30     4910.53     8604.00        0.00      130.39       68.20
#    150          32      163.89     6060.13    14922.00        0.00      179.28       75.35
#    200          32      139.25     7133.42    10436.00        0.00      228.98       79.85


#
# R = 48
#

# build_disk_index <type> <data> <prefix> <R> <L> <PQ_bytes> <M_GB> <threads> <metric> <nbr_type>
build/tests/build_disk_index float data_gist.bin pipeann_gist_idx 48 150 32 64 32 l2 pq

# search_disk_index <type> <prefix> <threads> <beam_width> <query> <gt> <topk> <metric> <nbr_type> <mode> <mem_L> <Ls...>
build/tests/search_disk_index float pipeann_gist_idx 1 32 query_gist.bin gt_gist.bin 10 l2 pq 2 0 50 100 150 200

#
# Example output:
#
#      L   I/O Width         QPS  AvgLat(us)     P99 Lat   Mean Hops    Mean IOs   Recall@10
# =========================================================================================
#     50          32      292.21     3392.56    17332.00        0.00       76.15       56.72
#    100          32      209.99     4721.30     6935.00        0.00      124.36       69.06
#    150          32      167.44     5922.12    15450.00        0.00      173.41       75.60
#    200          32      143.46     6902.39    16531.00        0.00      222.74       80.05
