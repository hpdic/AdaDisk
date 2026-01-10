# 1. 创建目录
mkdir -p ~/hpdic/gist_data
cd ~/hpdic/gist_data

# 2. 下载 GIST1M (约 2.7GB)
# 使用 ftp.irisa.fr 源，如果太慢可以尝试找别的镜像，但这通常是最稳的
echo "🚀 开始下载 GIST1M..."
wget -c ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz

# 3. 解压
echo "📦 正在解压..."
tar -zxvf gist.tar.gz

# 4. 整理文件 (解压出来通常在一个 gist 文件夹里，我们把它移出来)
mv gist/* .
rmdir gist

echo "✅ GIST1M 准备完毕！"
ls -lh