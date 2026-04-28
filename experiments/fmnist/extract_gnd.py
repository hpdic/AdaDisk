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
