#!/bin/bash

DISKANN_HOME=${HOME}/hpdic/AdaDisk
SEARCH_BIN=${DISKANN_HOME}/build/apps/search_disk_index

INDEX_PREFIX=${HOME}/hpdic/data_fmnist/indices_mcgi/diskann_mcgi_R32_L50_B2G
QUERY_FILE=${HOME}/hpdic/data_fmnist/fashion_query.bin
GT_FILE=${HOME}/hpdic/data_fmnist/fashion_gnd.bin

RESULT_OUTPUT=search_results_fmnist_mcgi.bin

K=10
L_LIST='10 20 40 80 100 120 140 160 180 200'
THREADS=96

echo 'Start searching fmnist mcgi...'

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


# (venv) cc@uc-ssd:~/hpdic/AdaDisk/experiments/fmnist$ bash fmnist_search_mcgi.sh 
# Start searching fmnist mcgi...
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

# Reading bin file /home/cc/hpdic/data_fmnist/indices_mcgi/diskann_mcgi_R32_L50_B2G_pq_compressed.bin ...
# Opening bin file /home/cc/hpdic/data_fmnist/indices_mcgi/diskann_mcgi_R32_L50_B2G_pq_compressed.bin... 
# Metadata: #pts = 60000, #dims = 512...
# done.
# Reading bin file /home/cc/hpdic/data_fmnist/indices_mcgi/diskann_mcgi_R32_L50_B2G_pq_pivots.bin ...
# Opening bin file /home/cc/hpdic/data_fmnist/indices_mcgi/diskann_mcgi_R32_L50_B2G_pq_pivots.bin... 
# Metadata: #pts = 4, #dims = 1...
# done.
# Offsets: 4096 806920 810064 812124
# Reading bin file /home/cc/hpdic/data_fmnist/indices_mcgi/diskann_mcgi_R32_L50_B2G_pq_pivots.bin ...
# Opening bin file /home/cc/hpdic/data_fmnist/indices_mcgi/diskann_mcgi_R32_L50_B2G_pq_pivots.bin... 
# Metadata: #pts = 256, #dims = 784...
# done.
# Reading bin file /home/cc/hpdic/data_fmnist/indices_mcgi/diskann_mcgi_R32_L50_B2G_pq_pivots.bin ...
# Opening bin file /home/cc/hpdic/data_fmnist/indices_mcgi/diskann_mcgi_R32_L50_B2G_pq_pivots.bin... 
# Metadata: #pts = 784, #dims = 1...
# done.
# Reading bin file /home/cc/hpdic/data_fmnist/indices_mcgi/diskann_mcgi_R32_L50_B2G_pq_pivots.bin ...
# Opening bin file /home/cc/hpdic/data_fmnist/indices_mcgi/diskann_mcgi_R32_L50_B2G_pq_pivots.bin... 
# Metadata: #pts = 513, #dims = 1...
# done.
# Loaded PQ Pivots: #ctrs: 256, #dims: 784, #chunks: 512
# Loaded PQ centroids and in-memory compressed vectors. #points: 60000 #dim: 784 #aligned_dim: 784 #chunks: 512
# Disk-Index File Meta-data: # nodes per sector: 1, max node len (bytes): 3268, max node degree: 32
# Opened file : /home/cc/hpdic/data_fmnist/indices_mcgi/diskann_mcgi_R32_L50_B2G_disk.index
# Setting up thread-specific contexts for nthreads: 96
# allocating ctx: 0x768764e8d000 to thread-id:130314114139840
# allocating ctx: 0x768764b8a000 to thread-id:130314189674176
# allocating ctx: 0x768764b79000 to thread-id:130314668058304
# allocating ctx: 0x768764b68000 to thread-id:130314139317952
# allocating ctx: 0x768764b57000 to thread-id:130314088961728
# allocating ctx: 0x768764b46000 to thread-id:130314198066880
# allocating ctx: 0x768764b35000 to thread-id:130314097354432
# allocating ctx: 0x768764b24000 to thread-id:130314449848000
# allocating ctx: 0x768764b13000 to thread-id:130314105747136
# allocating ctx: 0x768764b02000 to thread-id:130314080569024
# allocating ctx: 0x768764af1000 to thread-id:130314172888768
# allocating ctx: 0x768764ae0000 to thread-id:130314382706368
# allocating ctx: 0x768764acf000 to thread-id:130314483418816
# allocating ctx: 0x768764abe000 to thread-id:130314164496064
# allocating ctx: 0x768764aad000 to thread-id:130314214852288
# allocating ctx: 0x768764a9c000 to thread-id:130314130925248
# allocating ctx: 0x768764a8b000 to thread-id:130314147710656
# allocating ctx: 0x768764a7a000 to thread-id:130314584131264
# allocating ctx: 0x768764a69000 to thread-id:130314181281472
# allocating ctx: 0x768764a58000 to thread-id:130314156103360
# allocating ctx: 0x768764a47000 to thread-id:130314206459584
# allocating ctx: 0x768764a36000 to thread-id:130314357528256
# allocating ctx: 0x768764a25000 to thread-id:130314433062592
# allocating ctx: 0x768764a14000 to thread-id:130314407884480
# allocating ctx: 0x7687647ef000 to thread-id:130314651272896
# allocating ctx: 0x7687647de000 to thread-id:130314391099072
# allocating ctx: 0x7687647cd000 to thread-id:130314567345856
# allocating ctx: 0x7687647bc000 to thread-id:130314500204224
# allocating ctx: 0x7687647ab000 to thread-id:130314365920960
# allocating ctx: 0x76876479a000 to thread-id:130314550560448
# allocating ctx: 0x768764789000 to thread-id:130314374313664
# allocating ctx: 0x768764778000 to thread-id:130314533775040
# allocating ctx: 0x768764767000 to thread-id:130314600916672
# allocating ctx: 0x768764756000 to thread-id:130314617702080
# allocating ctx: 0x768764745000 to thread-id:130314676451008
# allocating ctx: 0x768764734000 to thread-id:130314122532544
# allocating ctx: 0x768764723000 to thread-id:130314424669888
# allocating ctx: 0x768764712000 to thread-id:130314516989632
# allocating ctx: 0x768764701000 to thread-id:130313921107648
# allocating ctx: 0x7687646f0000 to thread-id:130314441455296
# allocating ctx: 0x7687646df000 to thread-id:130314072176320
# allocating ctx: 0x7687646ce000 to thread-id:130314298779328
# allocating ctx: 0x7687646bd000 to thread-id:130314273601216
# allocating ctx: 0x7687646ac000 to thread-id:130314223244992
# allocating ctx: 0x76876469b000 to thread-id:130314634487488
# allocating ctx: 0x76876468a000 to thread-id:130314013427392
# allocating ctx: 0x768764679000 to thread-id:130314005034688
# allocating ctx: 0x768764668000 to thread-id:130323897301056
# allocating ctx: 0x768764657000 to thread-id:130314340742848
# allocating ctx: 0x768764646000 to thread-id:130313929500352
# allocating ctx: 0x768764635000 to thread-id:130313988249280
# allocating ctx: 0x768764624000 to thread-id:130314265208512
# allocating ctx: 0x76876200f000 to thread-id:130314701629120
# allocating ctx: 0x768761ffe000 to thread-id:130313954678464
# allocating ctx: 0x768761fed000 to thread-id:130314021820096
# allocating ctx: 0x768761fdc000 to thread-id:130314592523968
# allocating ctx: 0x768761fcb000 to thread-id:130314323957440
# allocating ctx: 0x768761fba000 to thread-id:130314307172032
# allocating ctx: 0x768761fa9000 to thread-id:130314030212800
# allocating ctx: 0x768761f98000 to thread-id:130314458240704
# allocating ctx: 0x768761f87000 to thread-id:130314349135552
# allocating ctx: 0x768761f76000 to thread-id:130314466633408
# allocating ctx: 0x768761f65000 to thread-id:130313963071168
# allocating ctx: 0x768761f54000 to thread-id:130313937893056
# allocating ctx: 0x768761f43000 to thread-id:130314642880192
# allocating ctx: 0x768761f32000 to thread-id:130314693236416
# allocating ctx: 0x768761f21000 to thread-id:130314475026112
# allocating ctx: 0x768761f10000 to thread-id:130314063783616
# allocating ctx: 0x768761eff000 to thread-id:130314055390912
# allocating ctx: 0x768761eee000 to thread-id:130313996641984
# allocating ctx: 0x768761edd000 to thread-id:130314508596928
# allocating ctx: 0x768761ecc000 to thread-id:130314399491776
# allocating ctx: 0x768761ebb000 to thread-id:130314248423104
# allocating ctx: 0x768761eaa000 to thread-id:130314416277184
# allocating ctx: 0x768761e99000 to thread-id:130314558953152
# allocating ctx: 0x768761e88000 to thread-id:130314240030400
# allocating ctx: 0x7687617ef000 to thread-id:130313979856576
# allocating ctx: 0x7687617de000 to thread-id:130313946285760
# allocating ctx: 0x7687617cd000 to thread-id:130314281993920
# allocating ctx: 0x7687617bc000 to thread-id:130314046998208
# allocating ctx: 0x7687617ab000 to thread-id:130314659665600
# allocating ctx: 0x76876179a000 to thread-id:130314332350144
# allocating ctx: 0x768761789000 to thread-id:130314542167744
# allocating ctx: 0x768761778000 to thread-id:130314626094784
# allocating ctx: 0x768761767000 to thread-id:130314315564736
# allocating ctx: 0x768761756000 to thread-id:130314609309376
# allocating ctx: 0x768761745000 to thread-id:130314231637696
# allocating ctx: 0x768761734000 to thread-id:130314710021824
# allocating ctx: 0x768761723000 to thread-id:130314525382336
# allocating ctx: 0x7687603ed000 to thread-id:130314038605504
# allocating ctx: 0x7687603dc000 to thread-id:130314290386624
# allocating ctx: 0x7687603cb000 to thread-id:130314684843712
# allocating ctx: 0x7687603ba000 to thread-id:130314491811520
# allocating ctx: 0x7687603a9000 to thread-id:130313971463872
# allocating ctx: 0x768760398000 to thread-id:130314575738560
# allocating ctx: 0x768760387000 to thread-id:130314256815808
# Loading centroid data from medoids vector data of 1 medoid(s)
# done..
# Caching 10000 nodes around medoid(s)
# Reducing nodes to cache from: 10000 to: 6000(10 percent of total nodes:60000)
# Caching 6000...
# Level: 1.. #nodes: 1, #nodes thus far: 1
# Level: 2.. #nodes: 32, #nodes thus far: 33
# Level: 3.. #nodes: 655, #nodes thus far: 688
# Level: 4. #nodes: 5312, #nodes thus far: 6000
# done
# Loading the cache list into memory....done.
#      L   Beamwidth             QPS    Mean Latency    99.9 Latency        Mean IOs    Mean IO (us)         CPU (s)       Recall@10
# ===================================================================================================================================
#     10           2        10513.45         8991.55        43611.00           11.04         8619.96          319.02           96.02
#     20           2        27935.78         3291.35         9762.00           19.22         2669.68          534.42           99.35
#     40           2        16462.02         5676.20        10365.00           36.19         4638.86          888.83           99.83
#     80           2         8465.29        11154.13        34357.00           70.66         8834.19         2040.20           99.96
#    100           2         7063.46        13428.53        19020.00           87.97        11051.07         2051.04           99.97
#    120           2         5946.27        15958.41        23477.00          105.32        13148.48         2427.53           99.98
#    140           2         5098.55        18646.94        25257.00          122.66        15426.62         2780.26           99.98
#    160           2         4468.90        21284.24        27872.00          140.02        17623.54         3157.82           99.98
#    180           2         3968.96        23967.61        43033.00          157.43        19751.76         3648.06           99.99
#    200           2         3586.11        26553.14        42656.00          174.82        21875.60         4061.37           99.99
# Done searching. Now saving results 
# Writing bin: search_results_fmnist_mcgi.bin_10_idx_uint32.bin
# bin: #pts = 10000, #dims = 10, size = 400008B
# Finished writing bin.
# Writing bin: search_results_fmnist_mcgi.bin_10_dists_float.bin
# bin: #pts = 10000, #dims = 10, size = 400008B
# Finished writing bin.
# Writing bin: search_results_fmnist_mcgi.bin_20_idx_uint32.bin
# bin: #pts = 10000, #dims = 10, size = 400008B
# Finished writing bin.
# Writing bin: search_results_fmnist_mcgi.bin_20_dists_float.bin
# bin: #pts = 10000, #dims = 10, size = 400008B
# Finished writing bin.
# Writing bin: search_results_fmnist_mcgi.bin_40_idx_uint32.bin
# bin: #pts = 10000, #dims = 10, size = 400008B
# Finished writing bin.
# Writing bin: search_results_fmnist_mcgi.bin_40_dists_float.bin
# bin: #pts = 10000, #dims = 10, size = 400008B
# Finished writing bin.
# Writing bin: search_results_fmnist_mcgi.bin_80_idx_uint32.bin
# bin: #pts = 10000, #dims = 10, size = 400008B
# Finished writing bin.
# Writing bin: search_results_fmnist_mcgi.bin_80_dists_float.bin
# bin: #pts = 10000, #dims = 10, size = 400008B
# Finished writing bin.
# Writing bin: search_results_fmnist_mcgi.bin_100_idx_uint32.bin
# bin: #pts = 10000, #dims = 10, size = 400008B
# Finished writing bin.
# Writing bin: search_results_fmnist_mcgi.bin_100_dists_float.bin
# bin: #pts = 10000, #dims = 10, size = 400008B
# Finished writing bin.
# Writing bin: search_results_fmnist_mcgi.bin_120_idx_uint32.bin
# bin: #pts = 10000, #dims = 10, size = 400008B
# Finished writing bin.
# Writing bin: search_results_fmnist_mcgi.bin_120_dists_float.bin
# bin: #pts = 10000, #dims = 10, size = 400008B
# Finished writing bin.
# Writing bin: search_results_fmnist_mcgi.bin_140_idx_uint32.bin
# bin: #pts = 10000, #dims = 10, size = 400008B
# Finished writing bin.
# Writing bin: search_results_fmnist_mcgi.bin_140_dists_float.bin
# bin: #pts = 10000, #dims = 10, size = 400008B
# Finished writing bin.
# Writing bin: search_results_fmnist_mcgi.bin_160_idx_uint32.bin
# bin: #pts = 10000, #dims = 10, size = 400008B
# Finished writing bin.
# Writing bin: search_results_fmnist_mcgi.bin_160_dists_float.bin
# bin: #pts = 10000, #dims = 10, size = 400008B
# Finished writing bin.
# Writing bin: search_results_fmnist_mcgi.bin_180_idx_uint32.bin
# bin: #pts = 10000, #dims = 10, size = 400008B
# Finished writing bin.
# Writing bin: search_results_fmnist_mcgi.bin_180_dists_float.bin
# bin: #pts = 10000, #dims = 10, size = 400008B
# Finished writing bin.
# Writing bin: search_results_fmnist_mcgi.bin_200_idx_uint32.bin
# bin: #pts = 10000, #dims = 10, size = 400008B
# Finished writing bin.
# Writing bin: search_results_fmnist_mcgi.bin_200_dists_float.bin
# bin: #pts = 10000, #dims = 10, size = 400008B
# Finished writing bin.
# (venv) cc@uc-ssd:~/hpdic/AdaDisk/experiments/fmnist$   
# Clearing scratch