#!/bin/bash
# Clean QPS: stock DiskANN and MCGI (query-adaptive L, weights from final_eval.py) searched one
# at a time on an idle machine, page cache dropped before each run, one run per setting.
# Usage: clean_qps.sh   (results in ~/data/mcgi/clean/<ds>_<index>_{stock,mcgi}.log)
S=/home/cc/hpdic/adadisk/build/apps/search_disk_index
D=/home/cc/hpdic/adadisk/experiments/data
OUT=$HOME/data/mcgi/clean
mkdir -p $OUT
export TMPDIR=$HOME/data/mcgi/tmp
LS="10 15 20 25 30 40 50 60 80 100 150 200 300 400"
for pair in sift:diskann_R32_L50 glove:diskann_R64_L100 gist:diskann_R32_L100; do
  ds=${pair%%:*}; n=${pair#*:}
  AD=$(grep -a -o "MCGI_ADAPT=[^ ]*" $HOME/data/mcgi/final2/$ds/$n/adapt.txt | cut -d= -f2)
  ARGS="--data_type float --dist_fn l2 --index_path_prefix $HOME/data/mcgi/rl/$ds/$n/idx --query_file $D/$ds/${ds}_query.bin --gt_file $D/$ds/${ds}_gt.bin -K 10 -L $LS -T 32"
  for tag in stock mcgi; do
    sudo -n sh -c "echo 3 > /proc/sys/vm/drop_caches"
    if [ $tag = mcgi ]; then export MCGI_ADAPT=$AD; else unset MCGI_ADAPT; fi
    $S $ARGS --result_path $OUT/res > $OUT/${ds}_${n}_$tag.log 2>&1
    rm -f $OUT/res_*
  done
done
