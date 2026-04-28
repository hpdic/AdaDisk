#!/bin/bash

DISKANN_HOME=${HOME}/hpdic/AdaDisk
SEARCH_BIN=${DISKANN_HOME}/build/apps/search_disk_index

INDEX_PREFIX=${HOME}/hpdic/data_fmnist/indices/diskann_base_R32_L50_B2G
QUERY_FILE=${HOME}/hpdic/data_fmnist/fashion_query.bin
GT_FILE=${HOME}/hpdic/data_fmnist/fashion_gnd.bin

RESULT_OUTPUT=search_results_fmnist_base.bin

K=10
L_LIST='10 20 40 80 100 120 140 160 180 200'
THREADS=96

echo 'Start searching fmnist baseline...'

${SEARCH_BIN} \
  --data_type float \
  --dist_fn l2 \
  --index_path_prefix ${INDEX_PREFIX} \
  --query_file ${QUERY_FILE} \
  --gt_file ${GT_FILE} \
  -K ${K} \
  -L ${L_LIST} \
  --result_path ${RESULT_OUTPUT} \
  --num_nodes_to_cache 10000 \
  -T ${THREADS}


# (venv) cc@uc-ssd:~/hpdic/AdaDisk/experiments/fmnist$ 
# (venv) cc@uc-ssd:~/hpdic/AdaDisk/experiments/fmnist$ bash fmnist_search_baseline.sh 
# Start searching fmnist baseline...
# Search parameters: #threads: 96,  beamwidth: 2.
# Reading (with alignment) bin file /home/cc/hpdic/data_fmnist/fashion_query.bin ...Metadata: #pts = 10000, #dims = 784, aligned_dim = 784... allocating aligned memory of 31360000 bytes... done. Copying data to mem_aligned buffer... done.
# Opened: /home/cc/hpdic/data_fmnist/fashion_gnd.bin, size: 4000008, cache_size: 4000008
# Reading truthset file /home/cc/hpdic/data_fmnist/fashion_gnd.bin ...
# Metadata: #pts = 10000, #dims = 100... 
# L2: Using AVX2 distance computation DistanceL2Float
# L2: Using AVX2 distance computation DistanceL2Float


# *****************************************************
# * ==HPDIC MOD== Hello from modified libdiskann.so!  *
# * I am inside src/pq_flash_index.cpp :: load()      *
# *****************************************************

# Reading bin file /home/cc/hpdic/data_fmnist/indices/diskann_base_R32_L50_B2G_pq_compressed.bin ...
# Opening bin file /home/cc/hpdic/data_fmnist/indices/diskann_base_R32_L50_B2G_pq_compressed.bin... 
# Metadata: #pts = 60000, #dims = 512...
# done.
# Reading bin file /home/cc/hpdic/data_fmnist/indices/diskann_base_R32_L50_B2G_pq_pivots.bin ...
# Opening bin file /home/cc/hpdic/data_fmnist/indices/diskann_base_R32_L50_B2G_pq_pivots.bin... 
# Metadata: #pts = 4, #dims = 1...
# done.
# Offsets: 4096 806920 810064 812124
# Reading bin file /home/cc/hpdic/data_fmnist/indices/diskann_base_R32_L50_B2G_pq_pivots.bin ...
# Opening bin file /home/cc/hpdic/data_fmnist/indices/diskann_base_R32_L50_B2G_pq_pivots.bin... 
# Metadata: #pts = 256, #dims = 784...
# done.
# Reading bin file /home/cc/hpdic/data_fmnist/indices/diskann_base_R32_L50_B2G_pq_pivots.bin ...
# Opening bin file /home/cc/hpdic/data_fmnist/indices/diskann_base_R32_L50_B2G_pq_pivots.bin... 
# Metadata: #pts = 784, #dims = 1...
# done.
# Reading bin file /home/cc/hpdic/data_fmnist/indices/diskann_base_R32_L50_B2G_pq_pivots.bin ...
# Opening bin file /home/cc/hpdic/data_fmnist/indices/diskann_base_R32_L50_B2G_pq_pivots.bin... 
# Metadata: #pts = 513, #dims = 1...
# done.
# Loaded PQ Pivots: #ctrs: 256, #dims: 784, #chunks: 512
# Loaded PQ centroids and in-memory compressed vectors. #points: 60000 #dim: 784 #aligned_dim: 784 #chunks: 512
# Disk-Index File Meta-data: # nodes per sector: 1, max node len (bytes): 3268, max node degree: 32
# Opened file : /home/cc/hpdic/data_fmnist/indices/diskann_base_R32_L50_B2G_disk.index
# Setting up thread-specific contexts for nthreads: 96
# allocating ctx: 0x7c0c3079b000 to thread-id:136391794878528
# allocating ctx: 0x7c0c2fc8d000 to thread-id:136381860570816
# allocating ctx: 0x7c0c2f958000 to thread-id:136381843785408
# allocating ctx: 0x7c0c2f947000 to thread-id:136382095566528
# allocating ctx: 0x7c0c2f936000 to thread-id:136381852178112
# allocating ctx: 0x7c0c2f925000 to thread-id:136382020032192
# allocating ctx: 0x7c0c2f914000 to thread-id:136382003246784
# allocating ctx: 0x7c0c2f903000 to thread-id:136382036817600
# allocating ctx: 0x7c0c2f8f2000 to thread-id:136382028424896
# allocating ctx: 0x7c0c2f8e1000 to thread-id:136382070388416
# allocating ctx: 0x7c0c2f8d0000 to thread-id:136381902534336
# allocating ctx: 0x7c0c2f8bf000 to thread-id:136381919319744
# allocating ctx: 0x7c0c2f8ae000 to thread-id:136382380918464
# allocating ctx: 0x7c0c2f89d000 to thread-id:136381827000000
# allocating ctx: 0x7c0c2f88c000 to thread-id:136381818607296
# allocating ctx: 0x7c0c2f87b000 to thread-id:136381969675968
# allocating ctx: 0x7c0c2f86a000 to thread-id:136381986461376
# allocating ctx: 0x7c0c2f859000 to thread-id:136382305384128
# allocating ctx: 0x7c0c2f848000 to thread-id:136381877356224
# allocating ctx: 0x7c0c2f837000 to thread-id:136382599128768
# allocating ctx: 0x7c0c2f826000 to thread-id:136381885748928
# allocating ctx: 0x7c0c2f815000 to thread-id:136381910927040
# allocating ctx: 0x7c0c2f5ef000 to thread-id:136381952890560
# allocating ctx: 0x7c0c2f5de000 to thread-id:136382280206016
# allocating ctx: 0x7c0c2f5cd000 to thread-id:136382061995712
# allocating ctx: 0x7c0c2f5bc000 to thread-id:136382255027904
# allocating ctx: 0x7c0c2f5ab000 to thread-id:136382498416320
# allocating ctx: 0x7c0c2f59a000 to thread-id:136382397703872
# allocating ctx: 0x7c0c2f589000 to thread-id:136382523594432
# allocating ctx: 0x7c0c2f578000 to thread-id:136382288598720
# allocating ctx: 0x7c0c2f567000 to thread-id:136382431274688
# allocating ctx: 0x7c0c2f556000 to thread-id:136382112351936
# allocating ctx: 0x7c0c2f545000 to thread-id:136382213064384
# allocating ctx: 0x7c0c2f534000 to thread-id:136382322169536
# allocating ctx: 0x7c0c2f523000 to thread-id:136382515201728
# allocating ctx: 0x7c0c2f512000 to thread-id:136382347347648
# allocating ctx: 0x7c0c2f501000 to thread-id:136382296991424
# allocating ctx: 0x7c0c2f4f0000 to thread-id:136382263420608
# allocating ctx: 0x7c0c2f4df000 to thread-id:136382187886272
# allocating ctx: 0x7c0c2f4ce000 to thread-id:136382246635200
# allocating ctx: 0x7c0c2f4bd000 to thread-id:136382481630912
# allocating ctx: 0x7c0c2f4ac000 to thread-id:136382582343360
# allocating ctx: 0x7c0c2f49b000 to thread-id:136382355740352
# allocating ctx: 0x7c0c2f48a000 to thread-id:136382422881984
# allocating ctx: 0x7c0c2f479000 to thread-id:136382364133056
# allocating ctx: 0x7c0c2f468000 to thread-id:136382313776832
# allocating ctx: 0x7c0c2f457000 to thread-id:136381835392704
# allocating ctx: 0x7c0c2f446000 to thread-id:136382473238208
# allocating ctx: 0x7c0c2f435000 to thread-id:136382229849792
# allocating ctx: 0x7c0c2f424000 to thread-id:136382531987136
# allocating ctx: 0x7c0c2ce0f000 to thread-id:136382154315456
# allocating ctx: 0x7c0c2cdfe000 to thread-id:136382557165248
# allocating ctx: 0x7c0c2cded000 to thread-id:136382078781120
# allocating ctx: 0x7c0c2cddc000 to thread-id:136382590736064
# allocating ctx: 0x7c0c2cdcb000 to thread-id:136381978068672
# allocating ctx: 0x7c0c2cdba000 to thread-id:136382607521472
# allocating ctx: 0x7c0c2cda9000 to thread-id:136382145922752
# allocating ctx: 0x7c0c2cd98000 to thread-id:136382506809024
# allocating ctx: 0x7c0c2cd87000 to thread-id:136382238242496
# allocating ctx: 0x7c0c2cd76000 to thread-id:136382011639488
# allocating ctx: 0x7c0c2cd65000 to thread-id:136382087173824
# allocating ctx: 0x7c0c2cd54000 to thread-id:136382196278976
# allocating ctx: 0x7c0c2cd43000 to thread-id:136382338954944
# allocating ctx: 0x7c0c2cd32000 to thread-id:136382120744640
# allocating ctx: 0x7c0c2cd21000 to thread-id:136382439667392
# allocating ctx: 0x7c0c2cd10000 to thread-id:136382330562240
# allocating ctx: 0x7c0c2ccff000 to thread-id:136382103959232
# allocating ctx: 0x7c0c2ccee000 to thread-id:136382271813312
# allocating ctx: 0x7c0c2ccdd000 to thread-id:136381927712448
# allocating ctx: 0x7c0c2cccc000 to thread-id:136381961283264
# allocating ctx: 0x7c0c2ccbb000 to thread-id:136382129137344
# allocating ctx: 0x7c0c2ccaa000 to thread-id:136382406096576
# allocating ctx: 0x7c0c2cc99000 to thread-id:136382053603008
# allocating ctx: 0x7c0c2cc88000 to thread-id:136382414489280
# allocating ctx: 0x7c0c2c5ef000 to thread-id:136382372525760
# allocating ctx: 0x7c0c2c5de000 to thread-id:136381868963520
# allocating ctx: 0x7c0c2c5cd000 to thread-id:136382490023616
# allocating ctx: 0x7c0c2c5bc000 to thread-id:136381894141632
# allocating ctx: 0x7c0c2c5ab000 to thread-id:136382464845504
# allocating ctx: 0x7c0c2c59a000 to thread-id:136382137530048
# allocating ctx: 0x7c0c2c589000 to thread-id:136382456452800
# allocating ctx: 0x7c0c2c578000 to thread-id:136382540379840
# allocating ctx: 0x7c0c2c567000 to thread-id:136382565557952
# allocating ctx: 0x7c0c2c556000 to thread-id:136382221457088
# allocating ctx: 0x7c0c2c545000 to thread-id:136382204671680
# allocating ctx: 0x7c0c2c534000 to thread-id:136381944497856
# allocating ctx: 0x7c0c2c523000 to thread-id:136382179493568
# allocating ctx: 0x7c0c2b1ed000 to thread-id:136381936105152
# allocating ctx: 0x7c0c2b1dc000 to thread-id:136382573950656
# allocating ctx: 0x7c0c2b1cb000 to thread-id:136382389311168
# allocating ctx: 0x7c0c2b1ba000 to thread-id:136381994854080
# allocating ctx: 0x7c0c2b1a9000 to thread-id:136382162708160
# allocating ctx: 0x7c0c2b198000 to thread-id:136382448060096
# allocating ctx: 0x7c0c2b187000 to thread-id:136382548772544
# allocating ctx: 0x7c0c2b176000 to thread-id:136382045210304
# allocating ctx: 0x7c0c2b165000 to thread-id:136382171100864
# Loading centroid data from medoids vector data of 1 medoid(s)
# done..
# Caching 10000 nodes around medoid(s)
# Reducing nodes to cache from: 10000 to: 6000(10 percent of total nodes:60000)
# Caching 6000...
# Level: 1.. #nodes: 1, #nodes thus far: 1
# Level: 2.. #nodes: 32, #nodes thus far: 33
# Level: 3.. #nodes: 669, #nodes thus far: 702
# Level: 4. #nodes: 5298, #nodes thus far: 6000
# done
# Loading the cache list into memory....done.
#      L   Beamwidth             QPS    Mean Latency    99.9 Latency        Mean IOs    Mean IO (us)         CPU (s)       Recall@10
# ===================================================================================================================================
#     10           2        58842.52         1493.73         3692.00           10.72         1032.17          398.20           95.98
#     20           2        36041.31         2527.57         5187.00           18.91         1893.36          546.08           99.35
#     40           2        19826.77         4706.60         7969.00           35.86         3682.09          881.62           99.82
#     80           2        10099.27         9329.05        31403.00           70.29         7260.51         1800.80           99.95
#    100           2         8343.13        11358.59        18292.00           87.56         9168.14         1886.51           99.96
#    120           2         6972.93        13594.81        22152.00          104.88        10980.22         2254.07           99.98
#    140           2         6005.77        15818.21        20997.00          122.20        12914.82         2489.43           99.98
#    160           2         5237.07        18153.55        26235.00          139.54        14743.40         2934.83           99.98
#    180           2         4670.25        20362.60        27985.00          156.88        16536.01         3297.18           99.99
#    200           2         4161.97        22856.29        43401.00          174.22        18063.20         4204.65           99.99
# Done searching. Now saving results 
# Writing bin: search_results_fmnist_base.bin_10_idx_uint32.bin
# bin: #pts = 10000, #dims = 10, size = 400008B
# Finished writing bin.
# Writing bin: search_results_fmnist_base.bin_10_dists_float.bin
# bin: #pts = 10000, #dims = 10, size = 400008B
# Finished writing bin.
# Writing bin: search_results_fmnist_base.bin_20_idx_uint32.bin
# bin: #pts = 10000, #dims = 10, size = 400008B
# Finished writing bin.
# Writing bin: search_results_fmnist_base.bin_20_dists_float.bin
# bin: #pts = 10000, #dims = 10, size = 400008B
# Finished writing bin.
# Writing bin: search_results_fmnist_base.bin_40_idx_uint32.bin
# bin: #pts = 10000, #dims = 10, size = 400008B
# Finished writing bin.
# Writing bin: search_results_fmnist_base.bin_40_dists_float.bin
# bin: #pts = 10000, #dims = 10, size = 400008B
# Finished writing bin.
# Writing bin: search_results_fmnist_base.bin_80_idx_uint32.bin
# bin: #pts = 10000, #dims = 10, size = 400008B
# Finished writing bin.
# Writing bin: search_results_fmnist_base.bin_80_dists_float.bin
# bin: #pts = 10000, #dims = 10, size = 400008B
# Finished writing bin.
# Writing bin: search_results_fmnist_base.bin_100_idx_uint32.bin
# bin: #pts = 10000, #dims = 10, size = 400008B
# Finished writing bin.
# Writing bin: search_results_fmnist_base.bin_100_dists_float.bin
# bin: #pts = 10000, #dims = 10, size = 400008B
# Finished writing bin.
# Writing bin: search_results_fmnist_base.bin_120_idx_uint32.bin
# bin: #pts = 10000, #dims = 10, size = 400008B
# Finished writing bin.
# Writing bin: search_results_fmnist_base.bin_120_dists_float.bin
# bin: #pts = 10000, #dims = 10, size = 400008B
# Finished writing bin.
# Writing bin: search_results_fmnist_base.bin_140_idx_uint32.bin
# bin: #pts = 10000, #dims = 10, size = 400008B
# Finished writing bin.
# Writing bin: search_results_fmnist_base.bin_140_dists_float.bin
# bin: #pts = 10000, #dims = 10, size = 400008B
# Finished writing bin.
# Writing bin: search_results_fmnist_base.bin_160_idx_uint32.bin
# bin: #pts = 10000, #dims = 10, size = 400008B
# Finished writing bin.
# Writing bin: search_results_fmnist_base.bin_160_dists_float.bin
# bin: #pts = 10000, #dims = 10, size = 400008B
# Finished writing bin.
# Writing bin: search_results_fmnist_base.bin_180_idx_uint32.bin
# bin: #pts = 10000, #dims = 10, size = 400008B
# Finished writing bin.
# Writing bin: search_results_fmnist_base.bin_180_dists_float.bin
# bin: #pts = 10000, #dims = 10, size = 400008B
# Finished writing bin.
# Writing bin: search_results_fmnist_base.bin_200_idx_uint32.bin
# bin: #pts = 10000, #dims = 10, size = 400008B
# Finished writing bin.
# Writing bin: search_results_fmnist_base.bin_200_dists_float.bin
# bin: #pts = 10000, #dims = 10, size = 400008B
# Finished writing bin.
# Clearing scratch
# (venv) cc@uc-ssd:~/hpdic/AdaDisk/experiments/fmnist$ 