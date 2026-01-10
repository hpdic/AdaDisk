import os
import requests
import h5py
import numpy as np
import struct

# === 靠谱的 AWS S3 源 (1M Subset) ===
URL = "http://ann-benchmarks.com/deep-image-96-angular.hdf5"
HDF5_FILE = os.path.expanduser("~/deep1b_data/deep-image-96-angular.hdf5")
FBIN_FILE = os.path.expanduser("~/deep1b_data/deep1b_base_1M.fbin")

def download():
    if os.path.exists(HDF5_FILE):
        print(f"✅ {HDF5_FILE} 已存在，跳过下载。")
        return

    print(f"🚀 正在从 AWS 镜像下载 Deep1B (1M Subset)...")
    try:
        response = requests.get(URL, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(HDF5_FILE, "wb") as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        print(f"进度: {downloaded / 1024 / 1024:.1f} MB / {total_size / 1024 / 1024:.1f} MB", end='\r')
        print("\n✅ 下载完成！")
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        exit(1)

def convert():
    print(f"🔄 正在转换 HDF5 -> FBIN (DiskANN格式)...")
    try:
        f = h5py.File(HDF5_FILE, 'r')
        
        # ann-benchmarks 的数据通常在 'train' 键下
        data = f['train'][:]
        num, dim = data.shape
        print(f"数据形状: {num} vectors, {dim} dimensions")

        # 写入 .fbin 头 (num, dim)
        with open(FBIN_FILE, "wb") as out:
            header = struct.pack('ii', num, dim)
            out.write(header)
            # 写入数据 (float32)
            out.write(data.astype(np.float32).tobytes())
            
        print(f"✅ 转换成功: {FBIN_FILE}")
        
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        exit(1)

if __name__ == "__main__":
    download()
    convert()