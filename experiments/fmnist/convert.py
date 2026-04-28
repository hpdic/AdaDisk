import h5py
import struct
import numpy as np
import os

path_data = os.path.expanduser('~/hpdic/data_fmnist')

f = h5py.File(os.path.join(path_data, 'fashion-mnist-784-euclidean.hdf5'), 'r')

train_data = f['train'][:]
test_data = f['test'][:]

with open(os.path.join(path_data, 'fashion_base.bin'), 'wb') as fout:
    fout.write(struct.pack('<ii', train_data.shape[0], train_data.shape[1]))
    fout.write(train_data.tobytes())

with open(os.path.join(path_data, 'fashion_query.bin'), 'wb') as fout:
    fout.write(struct.pack('<ii', test_data.shape[0], test_data.shape[1]))
    fout.write(test_data.tobytes())

print('Convert finished')