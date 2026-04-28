cd ~/hpdic/AdaDisk/experiments/fmnist

pip3 install h5py numpy
mkdir -p ~/hpdic/data_fmnist
wget -O ~/hpdic/data_fmnist/fashion-mnist-784-euclidean.hdf5 http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5
python convert.py

echo 1048576 | sudo tee /proc/sys/fs/aio-max-nr
cat << 'EOF' > extract_gnd.py
import h5py
import struct
import numpy as np

hdf5_path = '/home/cc/hpdic/data_fmnist/fashion-mnist-784-euclidean.hdf5'
gnd_path = '/home/cc/hpdic/data_fmnist/fashion_gnd.bin'

print('Extracting Ground Truth...')
f = h5py.File(hdf5_path, 'r')
neighbors = f['neighbors'][:]

num_queries, num_gt = neighbors.shape

with open(gnd_path, 'wb') as fout:
    fout.write(struct.pack('<ii', num_queries, num_gt))
    fout.write(neighbors.astype(np.uint32).tobytes())

print('Convert finished: fashion_gnd.bin')
EOF
python3 extract_gnd.py

python fmnist_build_baseline.sh
python fminst_search_baseline.sh
python fmnist_build_mcgi.sh
python fmnist_search_mcgi.sh
    