import requests

# 使用 Google 的 DNS-over-HTTPS API，这走的是 HTTP 协议，防火墙一般不管
doh_url = "https://dns.google/resolve"
params = {
    "name": "bigannbenchmarks.blob.core.windows.net",
    "type": "A"
}

print("🔍 正在通过 Google DoH 获取最新 IP...")
try:
    r = requests.get(doh_url, params=params, timeout=5)
    data = r.json()
    
    if "Answer" in data:
        print("\n✅ 找到可用 IP (按优先级排序):")
        for ans in data["Answer"]:
            # type 1 是 A 记录 (IPv4)
            if ans["type"] == 1:
                print(f"   {ans['data']}")
        
        # 自动生成命令
        best_ip = data["Answer"][-1]["data"] # 通常最后一个是最终 CNAME 解析到的 IP
        print(f"\n🚀 建议修复命令 (复制执行):")
        print(f'echo "{best_ip} bigannbenchmarks.blob.core.windows.net" | sudo tee -a /etc/hosts')
    else:
        print("❌ 没有找到 A 记录，Azure 可能屏蔽了该区域。")
        print(data)

except Exception as e:
    print(f"❌ 连接 Google DNS 失败: {e}")
    # 备选 Cloudflare
    print("🔄 尝试 Cloudflare DoH...")
    try:
        r = requests.get("https://cloudflare-dns.com/dns-query", 
                         headers={"Accept": "application/dns-json"},
                         params={"name": "bigannbenchmarks.blob.core.windows.net", "type": "A"})
        data = r.json()
        if "Answer" in data:
            best_ip = data["Answer"][0]["data"]
            print(f"   {best_ip}")
            print(f"\n🚀 建议修复命令 (复制执行):")
            print(f'echo "{best_ip} bigannbenchmarks.blob.core.windows.net" | sudo tee -a /etc/hosts')
    except Exception as e2:
        print(f"❌ 彻底失败: {e2}")