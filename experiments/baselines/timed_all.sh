#!/bin/bash
# Timed phase: every measurement runs alone on the idle machine, one run per setting, page cache dropped first.
#  1. query-side ablation of MCGI on the three million-scale indexes (m1_tune/ablate.py run)
#  2. GIST1M R48 index: DiskANN and MCGI (weights from final2), same protocol as clean_qps.sh
#  3. PipeANN (pipelined search, mode 2) on the same GIST1M R32 / R48 DiskANN indexes, I/O width 4 and 32
#  4. CSPG (official library) on GIST1M, R32 / R48: build with 192 threads, then search with 32 threads
#  5. HNSW and IVF-Flat on GIST1M (memory_gist.py run)
set -e
export TMPDIR=$HOME/data/mcgi/tmp
M1=$HOME/hpdic/adadisk/experiments/m1_tune
BL=$HOME/hpdic/adadisk/experiments/baselines
D=$HOME/hpdic/adadisk/experiments/data/gist
S=$HOME/hpdic/adadisk/build/apps/search_disk_index
P=$HOME/hpdic/mcgi_baselines/PipeANN/build/tests/search_disk_index
LS="10 15 20 25 30 40 50 60 80 100 150 200 300 400"
drop() { sudo -n sh -c "echo 3 > /proc/sys/vm/drop_caches"; }

cd $M1 && /home/cc/venvs/mcgi/bin/python ablate.py run

OUT=$HOME/data/mcgi/clean
n=diskann_R48_L100
AD=$(grep -a -o "MCGI_ADAPT=[^ ]*" $HOME/data/mcgi/final2/gist/$n/adapt.txt | cut -d= -f2)
ARGS="--data_type float --dist_fn l2 --index_path_prefix $HOME/data/mcgi/rl/gist/$n/idx --query_file $D/gist_query.bin --gt_file $D/gist_gt.bin -K 10 -L $LS -T 32"
for tag in stock mcgi; do
  [ -s $OUT/gist_${n}_$tag.log ] && grep -aq "Done searching" $OUT/gist_${n}_$tag.log && continue
  drop
  if [ $tag = mcgi ]; then export MCGI_ADAPT=$AD; else unset MCGI_ADAPT; fi
  $S $ARGS --result_path $OUT/res > $OUT/gist_${n}_$tag.log 2>&1
  rm -f $OUT/res_*
done
unset MCGI_ADAPT
echo "gist R48 done"

for R in 32 48; do
  cd $HOME/data/mcgi/pipeann/gist_R$R
  for W in 4 32; do
    [ -s pipeann_W$W.log ] && grep -aq "Lock table" pipeann_W$W.log && continue
    drop
    $P float idx 32 $W $D/gist_query.bin $D/gist_gt.bin 10 l2 pq 2 10 $LS > pipeann_W$W.log 2>&1
  done
done
echo "pipeann done"

mkdir -p $HOME/data/mcgi/cspg
for R in 32 48; do
  [ -s $HOME/data/mcgi/cspg/cspg_R$R.csv ] && continue
  drop
  $HOME/hpdic/mcgi_baselines/cspg_gist $D/gist_base.fvecs $D/gist_query.fvecs $D/gist_groundtruth.ivecs $R 192 32 \
    $HOME/data/mcgi/cspg/cspg_R$R.csv > $HOME/data/mcgi/cspg/cspg_R$R.log 2>&1
done
echo "cspg done"

drop
cd $BL && $HOME/venvs/baselines/bin/python memory_gist.py run
echo "ALL DONE"
