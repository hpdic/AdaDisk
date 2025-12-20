"""
@file: gen_lid.py
@author: Dongfang Zhao
@email: dzhao@uw.edu
@brief: Generates synthetic LID (Local Intrinsic Dimensionality) data for MCGI testing.
        It mimics the binary format required by DiskANN.
"""

import numpy as np
import struct
import os

# 配置路径 (需与 shell 脚本一致)
DATA_DIR = "hpdic_data"
RAW_FILE = os.path.join(DATA_DIR, "ingest_raw.bin")
LID_FILE = os.path.join(DATA_DIR, "ingest_lid.bin")

def get_metadata(filepath):
    """读取 DiskANN bin 文件的头信息 (npts, dim)"""
    with open(filepath, "rb") as f:
        npts = struct.unpack("i", f.read(4))[0]
        dim = struct.unpack("i", f.read(4))[0]
    return npts, dim

def generate_synthetic_lid(npts):
    """
    生成模拟的 LID 分布。
    通常真实数据的 LID 类似正态分布或长尾分布。
    这里我们生成均值为 5.0，标准差为 2.0 的数据。
    """
    print(f"Generating synthetic LID for {npts} points...")
    print("Distribution: Normal(loc=5.0, scale=2.0)")
    
    # 生成数据
    lid_data = np.random.normal(loc=5.0, scale=2.0, size=npts)
    
    # 修正负数 (维度不可能是负的)
    lid_data = np.maximum(lid_data, 1.0) 
    
    # 转换为 float32 (C++ 默认 float)
    return lid_data.astype(np.float32)

def save_bin(filename, data):
    """
    保存为 DiskANN 标准二进制格式:
    [num_points (int32)] [dim (int32)] [data (float32 array)...]
    """
    npts = len(data)
    dim = 1  # LID 是标量，维度为 1
    
    print(f"Saving to {filename}...")
    with open(filename, "wb") as f:
        f.write(struct.pack("i", npts))
        f.write(struct.pack("i", dim))
        f.write(data.tobytes())
    print("Done.")

def main():
    if not os.path.exists(RAW_FILE):
        print(f"Error: Raw data file not found: {RAW_FILE}")
        print("Please run gen_data.py first.")
        return

    # 1. 读取原始数据的点数，保证一一对应
    npts, _ = get_metadata(RAW_FILE)
    print(f"Detected {npts} points in raw data.")

    # 2. 生成 LID 数据
    lid_data = generate_synthetic_lid(npts)

    # 3. 保存
    save_bin(LID_FILE, lid_data)

if __name__ == "__main__":
    main()