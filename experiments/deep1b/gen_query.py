import numpy as np
import struct

# 你的 10M 文件
BASE_FILE = "/home/cc/hpdic/deep1b_data/deep1b_base_1M.fbin"
QUERY_FILE = "/home/cc/hpdic/deep1b_data/deep1b_query.fbin"
DIM = 96
NUM_QUERIES = 10000

print(f"正在从 Base 文件提取 {NUM_QUERIES} 个向量作为 Query...")

with open(BASE_FILE, "rb") as f_in:
    # 跳过原文件的 8 字节头
    f_in.seek(8)
    # 读取数据
    data = np.fromfile(f_in, dtype=np.float32, count=NUM_QUERIES * DIM)
    data = data.reshape(NUM_QUERIES, DIM)

with open(QUERY_FILE, "wb") as f_out:
    # 写入正确的 fbin 头: [n, d]
    f_out.write(struct.pack('ii', NUM_QUERIES, DIM))
    f_out.write(data.tobytes())

print(f"✅ 提取成功: {QUERY_FILE}")