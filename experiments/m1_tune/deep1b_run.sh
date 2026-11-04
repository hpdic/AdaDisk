#!/bin/bash
# Deep1B: one stock DiskANN index (L2, R32 L50, 31-byte PQ);
# MCGI = query-adaptive L on the same index. Fit on the even half of the 10K official queries (queries 449, 2382, 2809 excluded),
# clean serial searches of the odd half: stock, then MCGI.
D=$HOME/hpdic/datasets/deep1b; M=$HOME/data/mcgi/deep1b; I=$M/diskann_R32_L50/idx
S=/home/cc/hpdic/adadisk/build/apps/search_disk_index; B=/home/cc/hpdic/adadisk/build/apps/build_disk_index
export TMPDIR=$HOME/data/mcgi/tmp

[ "$(stat -c %s $D/base.1B.fbin)" = "384000000008" ] || { echo "download incomplete"; exit 1; }
mkdir -p $M/diskann_R32_L50 && cd $M/diskann_R32_L50
$B --data_type float --dist_fn l2 --data_path $D/base.1B.fbin --index_path_prefix $I -R 32 -L 50 -B 29.8 -M 300 -T 192 > build.log 2>&1 &
BP=$!
while kill -0 $BP 2>/dev/null; do   # memory guard: stop the build before the host runs out of RAM
  if [ "$(free -g | awk '/Mem:/{print $7}')" -lt 60 ]; then kill $BP; echo "killed: low memory" >> build.log; fi
  sudo -n dmesg | grep -q "does not have buffers attached" && { kill $BP; echo "killed: dmesg" >> build.log; }
  sleep 30
done
grep -aq "Indexing time" build.log || { echo "build failed"; exit 1; }
O=$M/eval; mkdir -p $O; cd $O
LS="10 15 20 25 30 40 50 60 80 100 150 200 300 400 500 600 800 1000"
sudo -n sh -c "echo 3 > /proc/sys/vm/drop_caches"
MCGI_FEAT_DUMP=1 MCGI_FEAT_N0=24 $S --data_type float --dist_fn l2 --index_path_prefix $I --query_file $M/query_even.fbin --gt_file $M/gt100_even.bin -K 10 -L $LS -W 2 -T 32 --result_path calib > calib.log 2>&1
AD=$(cd /home/cc/hpdic/adadisk/experiments/m1_tune && /home/cc/venvs/mcgi/bin/python -c "
import final_eval as fe
a, r2, g = fe.fit('$O', '$M/gt100_even.bin')
print(a)
print(f'R2={r2:.3f} gain={g:.3f}', file=open('$O/fit.txt', 'w'))")
echo "MCGI_ADAPT=$AD" >> fit.txt
for tag in stock mcgi; do
  sudo -n sh -c "echo 3 > /proc/sys/vm/drop_caches"
  if [ $tag = mcgi ]; then export MCGI_ADAPT=$AD; else unset MCGI_ADAPT; fi
  $S --data_type float --dist_fn l2 --index_path_prefix $I --query_file $M/query_odd.fbin --gt_file $M/gt100_odd.bin -K 10 -L $LS -W 2 -T 32 --result_path $tag > $tag.log 2>&1
  rm -f ${tag}_*
done
echo DONE >> fit.txt
