#!/bin/bash
# Extend the Deep1B clean serial curves to longer lists (same index, same fitted weights).
S=/home/cc/hpdic/adadisk/build/apps/search_disk_index
M=/home/cc/data/mcgi/deep1b; I=$M/diskann_R32_L50/idx; O=$M/eval; cd $O
AD=$(grep MCGI_ADAPT= fit.txt | cut -d= -f2)
LS="1200 1500 2000"
for tag in stock mcgi; do
  sudo -n sh -c "echo 3 > /proc/sys/vm/drop_caches"
  if [ $tag = mcgi ]; then export MCGI_ADAPT=$AD; else unset MCGI_ADAPT; fi
  $S --data_type float --dist_fn l2 --index_path_prefix $I --query_file $M/query_odd.fbin --gt_file $M/gt100_odd.bin -K 10 -L $LS -W 2 -T 32 --result_path ${tag}x > ${tag}_ext.log 2>&1
  rm -f ${tag}x_*
done
echo EXTDONE >> fit.txt
