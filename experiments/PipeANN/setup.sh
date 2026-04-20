cd ~/hpdic
git clone https://github.com/hpdic/PipeANN.git
cd PipeANN

cd third_party/liburing
./configure && make -j
cd ../..

bash ./build.sh  # Binaries in build/

# 转换 base 文件
build/tests/utils/vecs_to_bin float /home/cc/hpdic/AdaDisk/experiments/data/gist/gist_base.fvecs data_gist.bin

# 转换 query 文件
build/tests/utils/vecs_to_bin float /home/cc/hpdic/AdaDisk/experiments/data/gist/gist_query.fvecs query_gist.bin

# 转换 groundtruth 文件（注意 GT 通常是 int 类型）
build/tests/utils/vecs_to_bin int32 /home/cc/hpdic/AdaDisk/experiments/data/gist/gist_groundtruth.ivecs gt_gist.bin

####################################
############## R = 32 ##############
####################################

# build_disk_index <type> <data> <prefix> <R> <L> <PQ_bytes> <M_GB> <threads> <metric> <nbr_type>
build/tests/build_disk_index float data_gist.bin pipeann_gist_idx 32 150 32 64 32 l2 pq

# search_disk_index <type> <prefix> <threads> <beam_width> <query> <gt> <topk> <metric> <nbr_type> <mode> <mem_L> <Ls...>
build/tests/search_disk_index float pipeann_gist_idx 1 32 query_gist.bin gt_gist.bin 10 l2 pq 2 0 50 100 150 200 300 400 500 600 700 800 900 1000

#
# Example output:
#
#      L   I/O Width         QPS  AvgLat(us)     P99 Lat   Mean Hops    Mean IOs   Recall@10
# =========================================================================================
#     50          32      194.30     5118.52   241499.00        0.00       81.55       55.67
#    100          32      200.00     4967.51     8134.00        0.00      129.40       67.71
#    150          32      158.32     6270.23    14774.00        0.00      178.57       74.27
#    200          32      136.67     7268.91    10464.00        0.00      228.07       78.82
#    300          32      109.48     9068.43    13331.00        0.00      327.21       84.98
#    400          32       93.86    10576.26    16836.00        0.00      425.00       88.50
#    500          32       80.81    12284.63    18329.00        0.00      523.54       90.73
#    600          32       71.34    13919.91    18097.00        0.00      622.06       92.21
#    700          32       63.44    15640.68    19127.00        0.00      721.30       93.37
#    800          32       57.43    17274.72    22004.00        0.00      820.50       94.35
#    900          32       52.20    19008.69    24131.00        0.00      919.65       95.20
#   1000          32       47.94    20702.83    29039.00        0.00     1019.20       95.84


####################################
############## R = 48 ##############
####################################

# build_disk_index <type> <data> <prefix> <R> <L> <PQ_bytes> <M_GB> <threads> <metric> <nbr_type>
build/tests/build_disk_index float data_gist.bin pipeann_gist_idx 48 150 32 64 32 l2 pq

# search_disk_index <type> <prefix> <threads> <beam_width> <query> <gt> <topk> <metric> <nbr_type> <mode> <mem_L> <Ls...>
build/tests/search_disk_index float pipeann_gist_idx 1 32 query_gist.bin gt_gist.bin 10 l2 pq 2 0 50 100 150 200 300 400 500 600 700 800 900 1000

#
# Example output:
#
#      L   I/O Width         QPS  AvgLat(us)     P99 Lat   Mean Hops    Mean IOs   Recall@10
# =========================================================================================
#     50          32      293.62     3377.62    18501.00        0.00       75.97       56.70
#    100          32      214.78     4615.39     8028.00        0.00      123.86       68.48
#    150          32      170.83     5803.83    12210.00        0.00      172.94       75.41
#    200          32      145.97     6782.80    15105.00        0.00      222.46       79.92
#    300          32      117.35     8435.42    15526.00        0.00      321.76       85.68
#    400          32       98.51    10053.12    17425.00        0.00      419.96       89.20
#    500          32       84.83    11657.99    16892.00        0.00      518.71       91.24
#    600          32       74.38    13299.13    20560.00        0.00      617.79       92.89
#    700          32       66.21    14943.88    21951.00        0.00      717.10       94.00
#    800          32       60.25    16426.23    21920.00        0.00      816.52       94.91
#    900          32       54.75    18081.17    28998.00        0.00      915.94       95.74
#   1000          32       49.97    19819.80    26425.00        0.00     1015.53       96.36
