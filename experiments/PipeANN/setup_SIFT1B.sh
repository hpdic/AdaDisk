mkdir -p ~/hpdic/sift1b_data
cd ~/hpdic/sift1b_data

wget -c ftp://ftp.irisa.fr/local/texmex/corpus/bigann_query.bvecs.gz
wget -c ftp://ftp.irisa.fr/local/texmex/corpus/bigann_gnd.tar.gz
gzip -d bigann_query.bvecs.gz
tar -xvf bigann_gnd.tar.gz

nohup wget -c ftp://ftp.irisa.fr/local/texmex/corpus/bigann_base.bvecs.gz > download_base.log 2>&1 &
gzip -d bigann_base.bvecs.gz

cd ~/hpdic/PipeANN

# 转换十亿规模 base 文件
build/tests/utils/vecs_to_bin uint8 ~/hpdic/sift1b_data/bigann_base.bvecs data_sift1b.bin

# 转换 query 文件
build/tests/utils/vecs_to_bin uint8 ~/hpdic/sift1b_data/bigann_query.bvecs query_sift1b.bin

# 转换对应的十亿规模 groundtruth 文件
build/tests/utils/vecs_to_bin int32 ~/hpdic/sift1b_data/gnd/idx_1000M.ivecs gt_sift1b.bin

####################################
############## R = 32 ##############
####################################

# This is a very long process; consider tmux the following
tmux
mkdir -p ~/hpdic/sift1b_data

# build_disk_index <type> <data> <prefix> <R> <L> <PQ_bytes> <M_GB> <threads> <metric> <nbr_type>
build/tests/build_disk_index uint8 data_sift1b.bin ~/hpdic/sift1b_data/pipeann_sift1b_idx 32 50 16 150 96 l2 pq
# 10-11 hours for building the index on a single machine with 96 threads and 150 GB memory budget

# search_disk_index <type> <prefix> <threads> <beam_width> <query> <gt> <topk> <metric> <nbr_type> <mode> <mem_L> <Ls...>
build/tests/search_disk_index uint8 ~/hpdic/sift1b_data/pipeann_sift1b_idx 96 32 query_sift1b.bin gt_sift1b.bin 10 l2 pq 2 0 10 20 40 50 100 200 300 400 500 600 700 800 900 1000
#
# Example output:
#
#      L   I/O Width         QPS  AvgLat(us)     P99 Lat   Mean Hops    Mean IOs   Recall@10
# =========================================================================================
#     10          32     6766.91    13586.09   168000.00        0.00       69.12       28.60
#     20          32     5637.82    16607.12   220837.00        0.00       90.92       44.08
#     40          32     4903.37    19118.95   138655.00        0.00      131.73       58.99
#     50          32     4877.56    19266.78   125541.00        0.00      149.81       63.92
#    100          32     3828.11    24521.49   136833.00        0.00      201.93       75.30
#    200          32     2865.44    32886.08   166768.00        0.00      292.08       83.78
#    300          32     2181.98    43226.21   162943.00        0.00      382.89       87.72
#    400          32     1750.87    53766.08   256537.00        0.00      477.96       90.12
#    500          32     1527.03    61797.13   391809.00        0.00      576.79       91.77
#    600          32     1319.32    71773.05   281870.00        0.00      673.10       92.93
#    700          32     1152.65    82097.80   304806.00        0.00      771.34       93.81
#    800          32     1020.24    92829.08   345852.00        0.00      870.48       94.52
#    900          32      920.11   102942.02   391828.00        0.00      968.93       95.12
#   1000          32      831.31   114006.38   458459.00        0.00     1067.43       95.55
