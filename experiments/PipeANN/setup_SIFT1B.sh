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

mkdir -p ~/sift1b_data

# This is a very long process; consider tmux the following
tmux

# build_disk_index <type> <data> <prefix> <R> <L> <PQ_bytes> <M_GB> <threads> <metric> <nbr_type>
build/tests/build_disk_index uint8 data_sift1b.bin ~/sift1b_data/pipeann_sift1b_idx 32 50 16 300 256 l2 pq

# search_disk_index <type> <prefix> <threads> <beam_width> <query> <gt> <topk> <metric> <nbr_type> <mode> <mem_L> <Ls...>
build/tests/search_disk_index uint8 ~/sift1b_data/pipeann_sift1b_idx 256 32 query_sift1b.bin gt_sift1b.bin 10 l2 pq 2 0 10 20 40 50 100 200 300 400 500 600 700 800 900 1000

