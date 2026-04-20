# After installing PipeANN with the C++ interface, on the root directory of PipeANN, run the following commands to prepare the data for GIST1M experiments. Make sure to adjust the paths according to your setup.

# 转换 base 文件
build/tests/utils/vecs_to_bin float /home/cc/hpdic/AdaDisk/experiments/data/gist/gist_base.fvecs data_gist.bin

# 转换 query 文件
build/tests/utils/vecs_to_bin float /home/cc/hpdic/AdaDisk/experiments/data/gist/gist_query.fvecs query_gist.bin

# 转换 groundtruth 文件（注意 GT 通常是 int 类型）
build/tests/utils/vecs_to_bin int32 /home/cc/hpdic/AdaDisk/experiments/data/gist/gist_groundtruth.ivecs gt_gist.bin

# build_disk_index <type> <data> <prefix> <R> <L> <PQ_bytes> <M_GB> <threads> <metric> <nbr_type>
build/tests/build_disk_index float data_gist.bin pipeann_gist_idx 48 150 32 64 32 l2 pq