#!/bin/bash
# SIFT1B: one stock DiskANN index (R32 L50, 31-byte PQ); MCGI = query-adaptive L on the same index.
# 1. feature dump on the even official queries -> fit (final_eval.fit on the dump)
# 2. clean serial searches of the odd official queries: stock, then MCGI
M=$HOME/data/mcgi/sift1b; I=$M/diskann_R32_L50/idx; O=$M/eval; mkdir -p $O; cd $O
S=/home/cc/hpdic/adadisk/build/apps/search_disk_index
export TMPDIR=$HOME/data/mcgi/tmp
LS="10 15 20 25 30 40 50 60 80 100 150 200 300 400"
/home/cc/venvs/mcgi/bin/python - <<'PY'
import numpy as np, os
M=os.path.expanduser('~/data/mcgi/sift1b')
with open(f'{M}/query.u8bin','rb') as f:
    n,d=np.frombuffer(f.read(8),dtype=np.uint32); q=np.frombuffer(f.read(),dtype=np.uint8).reshape(n,d)[0::2]
with open(f'{M}/query_even.u8bin','wb') as f: f.write(np.array(q.shape,dtype=np.uint32).tobytes()); f.write(q.tobytes())
with open(f'{M}/gt100.bin','rb') as f:
    n,k=np.frombuffer(f.read(8),dtype=np.uint32); b=f.read()
ids=np.frombuffer(b[:n*k*4],dtype=np.uint32).reshape(n,k)[0::2]; ds=np.frombuffer(b[n*k*4:],dtype=np.float32).reshape(n,k)[0::2]
with open(f'{M}/gt100_even.bin','wb') as f: f.write(np.array(ids.shape,dtype=np.uint32).tobytes()); f.write(ids.tobytes()); f.write(ds.tobytes())
PY
sudo -n sh -c "echo 3 > /proc/sys/vm/drop_caches"
MCGI_FEAT_DUMP=1 MCGI_FEAT_N0=24 $S --data_type uint8 --dist_fn l2 --index_path_prefix $I --query_file $M/query_even.u8bin --gt_file $M/gt100_even.bin -K 10 -L $LS -W 2 -T 32 --result_path calib > calib.log 2>&1
AD=$(cd /home/cc/hpdic/adadisk/experiments/m1_tune && /home/cc/venvs/mcgi/bin/python -c "
import final_eval as fe
a, r2, g = fe.fit('$O', '$M/gt100_even.bin')
print(a)
print(f'R2={r2:.3f} gain={g:.3f}', file=open('$O/fit.txt', 'w'))")
echo "MCGI_ADAPT=$AD" >> fit.txt
for tag in stock mcgi; do
  sudo -n sh -c "echo 3 > /proc/sys/vm/drop_caches"
  if [ $tag = mcgi ]; then export MCGI_ADAPT=$AD; else unset MCGI_ADAPT; fi
  $S --data_type uint8 --dist_fn l2 --index_path_prefix $I --query_file $M/query_odd.u8bin --gt_file $M/gt100_odd.bin -K 10 -L $LS -W 2 -T 32 --result_path $tag > $tag.log 2>&1
  rm -f ${tag}_*
done
