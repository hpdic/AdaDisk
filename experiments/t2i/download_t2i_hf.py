import os
import requests
import struct
import numpy as np

# === HuggingFace 直链 (unum-cloud/ann-t2i-1m) ===
# 这是一个非常高质量的 T2I 1M 子集，200维
BASE_URL = "https://huggingface.co/datasets/unum-cloud/ann-t2i-1m/resolve/main/base.1M.fbin"
QUERY_URL = "https://huggingface.co/datasets/unum-cloud/ann-t2i-1m/resolve/main/query.public.100K.fbin"

# 本地文件名
BASE_FILE = os.path.expanduser("~/t2i_data/t2i_base_1M.fbin")
QUERY_FILE = os.path.expanduser("~/t2i_data/t2i_query.fbin")

def download_file(url, filename):
    if os.path.exists(filename):
        print(f"✅ {filename} 已存在，跳过。")
        return

    print(f"🚀 正在下载 {filename} ...")
    print(f"源: {url}")
    
    try:
        # stream=True 也就是流式下载
        with requests.get(url, stream=True, timeout=20) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            downloaded = 0
            
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): 
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0 and downloaded % (10*1024*1024) == 0:
                        print(f"进度: {downloaded/1024/1024:.1f} MB / {total_size/1024/1024:.1f} MB", end='\r')
        print(f"\n✅ 下载完成: {filename}")
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        print("💡 提示: 如果是网络超时，请尝试开启/关闭代理，或者设置 export HF_ENDPOINT=https://hf-mirror.com")
        exit(1)

def check_header(filename, expected_dim):
    with open(filename, 'rb') as f:
        num, dim = struct.unpack('ii', f.read(8))
        print(f"📄 校验 {filename}: Num={num}, Dim={dim}")
        if dim != expected_dim:
            print(f"⚠️ 警告: 维度不匹配! 预期 {expected_dim}, 实际 {dim}")

if __name__ == "__main__":
    # 1. 下载 Base (1M vectors, 200 dim)
    download_file(BASE_URL, BASE_FILE)
    check_header(BASE_FILE, 200)

    # 2. 下载 Query
    download_file(QUERY_URL, QUERY_FILE)
    check_header(QUERY_FILE, 200)
    
    print("\n🎉 数据准备完毕！")