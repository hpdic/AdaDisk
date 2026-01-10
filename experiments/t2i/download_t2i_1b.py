import os
import requests
import struct
import sys

# ================= 配置区域 =================
# T2I-1B 官方 Azure 源 (BigANN Challenge)
# 注意：这三个文件加起来大约 800GB，请确保磁盘空间充足
URLS = {
    "base": "https://bigannbenchmarks.blob.core.windows.net/data/t2i/base.1B.fbin",
    "query": "https://bigannbenchmarks.blob.core.windows.net/data/t2i/query.public.100K.fbin",
    "gt": "https://bigannbenchmarks.blob.core.windows.net/data/t2i/groundtruth.public.100K.bin"
}

# 本地存储路径 (根据你的习惯放在 hpdic 下)
DATA_DIR = os.path.expanduser("~/hpdic/t2i_data_1b")
FILES = {
    "base": os.path.join(DATA_DIR, "t2i_base_1B.fbin"),
    "query": os.path.join(DATA_DIR, "t2i_query.fbin"),
    "gt": os.path.join(DATA_DIR, "t2i_gt.bin")
}

def download_file_resumable(url, filename):
    # 确保目录存在
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # 获取本地已下载的大小
    existing_size = 0
    if os.path.exists(filename):
        existing_size = os.path.getsize(filename)
    
    #以此判断是否需要继续下载
    headers = {}
    if existing_size > 0:
        headers['Range'] = f'bytes={existing_size}-'
        print(f"🔄 检测到临时文件，尝试从 {existing_size/1024/1024/1024:.2f} GB 处断点续传...")

    try:
        # stream=True 开启流式下载
        with requests.get(url, stream=True, headers=headers, timeout=60) as r:
            # 416 表示 Range 请求范围错误（通常意味着已经下载完了）
            if r.status_code == 416:
                print(f"✅ {filename} 似乎已经下载完整。")
                return

            r.raise_for_status()
            
            # 获取本次请求的总大小 (注意：如果是续传，content-length 只是剩余部分的大小)
            total_size = int(r.headers.get('content-length', 0))
            if existing_size == 0:
                final_total_size = total_size
            else:
                # 尝试从 Content-Range 解析总大小 "bytes 1000-4999/5000"
                content_range = r.headers.get('Content-Range', '')
                if '/' in content_range:
                    final_total_size = int(content_range.split('/')[-1])
                else:
                    final_total_size = total_size + existing_size

            mode = 'ab' if existing_size > 0 else 'wb'
            downloaded_now = 0
            
            print(f"🚀 开始下载: {filename}")
            print(f"📏 总大小: {final_total_size/1024/1024/1024:.2f} GB")
            
            with open(filename, mode) as f:
                for chunk in r.iter_content(chunk_size=1024*1024): # 1MB chunk
                    if chunk:
                        f.write(chunk)
                        downloaded_now += len(chunk)
                        current_total = existing_size + downloaded_now
                        
                        # 打印进度条
                        if final_total_size > 0:
                            percent = (current_total / final_total_size) * 100
                            sys.stdout.write(f"\r进度: {percent:.2f}% | {current_total/1024/1024/1024:.2f} GB / {final_total_size/1024/1024/1024:.2f} GB")
                            sys.stdout.flush()
            
        print(f"\n✅ 下载完成: {filename}")
        
    except requests.exceptions.RequestException as e:
        print(f"\n❌ 下载中断: {e}")
        print("💡 提示: 请重新运行脚本，它会自动从断点处继续下载。")
        exit(1)

def check_header(filename, expected_num, expected_dim):
    if not os.path.exists(filename):
        return
        
    print(f"🔍 校验文件头: {filename}")
    try:
        with open(filename, 'rb') as f:
            # fbin 格式前 8 字节是 num (int32) 和 dim (int32)
            # gt bin 格式前 4 字节通常是 num
            header = f.read(8)
            num, dim = struct.unpack('ii', header)
            
            print(f"   -> 读出: Num={num}, Dim={dim}")
            
            if expected_num and num != expected_num:
                print(f"   ⚠️  警告: 点数不匹配! 预期 {expected_num}, 实际 {num}")
            if expected_dim and dim != expected_dim:
                print(f"   ⚠️  警告: 维度不匹配! 预期 {expected_dim}, 实际 {dim}")
            
            if expected_num == num and expected_dim == dim:
                print("   ✅ 校验通过")
    except Exception as e:
        print(f"   ❌ 校验失败: {e}")

if __name__ == "__main__":
    # 1. 下载 Query (比较小，先下)
    download_file_resumable(URLS["query"], FILES["query"])
    check_header(FILES["query"], 100000, 200)

    # 2. 下载 Ground Truth (直接下载，省去几天计算时间)
    download_file_resumable(URLS["gt"], FILES["gt"])
    # GT 文件的格式通常是：[num_queries, K, id1, id2, ..., idK, id1...] 或者是传统的 ivecs
    # 这里我们只简单下载，后续用脚本读取验证

    # 3. 下载 Base (800GB 大头)
    print("\n⚠️  准备下载 1B Base 数据 (约 800GB)，这可能需要很长时间...")
    download_file_resumable(URLS["base"], FILES["base"])
    check_header(FILES["base"], 1000000000, 200) # 预期 10亿点，200维
    
    print("\n🎉 T2I-1B 数据全量准备完毕！")