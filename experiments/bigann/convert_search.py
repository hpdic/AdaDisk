import numpy as np
import struct
import os

# --- 配置路径 ---
BASE_DIR = os.path.expanduser("~/hpdic/sift1b_data")
QUERY_IN  = os.path.join(BASE_DIR, "bigann_query.bvecs")
QUERY_OUT = os.path.join(BASE_DIR, "bigann_query.bin")

GT_IN  = os.path.join(BASE_DIR, "gnd/idx_1000M.ivecs")
GT_OUT = os.path.join(BASE_DIR, "bigann_gnd.bin")

def bvecs_to_bin(infile, outfile):
    print(f"🔄 Converting {infile} -> {outfile} ...")
    # bvecs: 每个向量是 (4字节维度 + 维度*1字节数据)
    # 读整个文件
    raw_data = np.fromfile(infile, dtype='uint8')
    
    # 读维度 (前4个字节)
    dim = raw_data[:4].view('int32')[0]
    print(f"   Detected Dimension: {dim}")
    
    # 计算行宽 (4字节头 + dim字节数据)
    row_bytes = 4 + dim
    num_points = raw_data.size // row_bytes
    print(f"   Detected Points: {num_points}")
    
    # 重塑数组
    reshaped = raw_data.reshape(num_points, row_bytes)
    
    # 扔掉每一行的前4个字节(维度头)，只留数据
    vectors = reshaped[:, 4:]
    
    # 写入 DiskANN 格式: [num_points(int)][dim(int)][data...]
    with open(outfile, 'wb') as f:
        f.write(struct.pack('I', num_points))
        f.write(struct.pack('I', dim))
        vectors.tofile(f)
    print("✅ Done.")

def ivecs_to_bin(infile, outfile):
    print(f"🔄 Converting {infile} -> {outfile} ...")
    # ivecs: 每个向量是 (4字节维度 + 维度*4字节int数据)
    # 按 int32 读取
    raw_data = np.fromfile(infile, dtype='int32')
    
    dim = raw_data[0]
    print(f"   Detected K (GT Neighbors): {dim}")
    
    row_ints = 1 + dim # 1个int头 + dim个int数据
    num_points = raw_data.size // row_ints
    print(f"   Detected Points: {num_points}")
    
    reshaped = raw_data.reshape(num_points, row_ints)
    
    # 扔掉头，转成 uint32 (DiskANN ID通常用unsigned)
    vectors = reshaped[:, 1:].astype('uint32')
    
    with open(outfile, 'wb') as f:
        f.write(struct.pack('I', num_points))
        f.write(struct.pack('I', dim))
        vectors.tofile(f)
    print("✅ Done.")

if __name__ == "__main__":
    if os.path.exists(QUERY_IN):
        bvecs_to_bin(QUERY_IN, QUERY_OUT)
    else:
        print(f"❌ Not Found: {QUERY_IN}")

    if os.path.exists(GT_IN):
        ivecs_to_bin(GT_IN, GT_OUT)
    else:
        print(f"❌ Not Found: {GT_IN}")