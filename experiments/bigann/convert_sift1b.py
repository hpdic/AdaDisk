import os
import struct
import numpy as np

# 配置
input_file = 'bigann_base.bvecs'
output_file = 'sift1b_base.bin'
dim = 128
# SIFT1B 是 10亿 (10^9)
num_points = 1000000000 

def convert_bvecs_to_bin():
    print(f"开始转换: {input_file} -> {output_file}")
    
    file_size = os.path.getsize(input_file)
    print(f"输入文件大小: {file_size / (1024**3):.2f} GB")
    
    # bvecs 每条记录: 4 bytes (header) + 128 bytes (uint8 data)
    record_size = 4 + dim
    
    # 简单的完整性检查
    if file_size != num_points * record_size:
        print(f"警告: 文件大小与预期不符！预期 {num_points * record_size}，实际 {file_size}")
    
    with open(input_file, 'rb') as f_in, open(output_file, 'wb') as f_out:
        # 1. 写入 DiskANN 格式的文件头: [num_points (int32), dim (int32)]
        header = struct.pack('ii', num_points, dim)
        f_out.write(header)
        
        # 2. 分块读取并写入
        # 为了速度，每次处理 1000 万条 (约 1.3GB)
        batch_count = 10000000 
        batch_bytes = batch_count * record_size
        
        processed = 0
        while processed < num_points:
            chunk = f_in.read(batch_bytes)
            if not chunk:
                break
            
            # 利用 numpy 快速去除每条记录前 4 个字节的 header
            # frombuffer 读取为 uint8 数组
            data_chunk = np.frombuffer(chunk, dtype=np.uint8)
            
            # reshape 成 (N, 132)
            # 因为每条是 4 bytes(header) + 128 bytes(data) = 132 bytes
            # 注意: 这里的 header 4 bytes 在 uint8 视角下是 4 个 uint8
            n_in_batch = len(chunk) // record_size
            data_reshaped = data_chunk.reshape(n_in_batch, record_size)
            
            # 切片，去掉前 4 列 (header)，保留后 128 列 (data)
            vectors = data_reshaped[:, 4:]
            
            # 写入纯数据
            f_out.write(vectors.tobytes())
            
            processed += n_in_batch
            if processed % 50000000 == 0:
                print(f"已处理: {processed / 100000000:.1f} 亿...")

    print("转换完成！")

if __name__ == '__main__':
    convert_bvecs_to_bin()