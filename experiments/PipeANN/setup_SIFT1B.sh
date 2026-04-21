mkdir -p ~/hpdic/sift1b_data
cd ~/hpdic/sift1b_data

wget -c ftp://ftp.irisa.fr/local/texmex/corpus/bigann_query.bvecs.gz
wget -c ftp://ftp.irisa.fr/local/texmex/corpus/bigann_gnd.tar.gz
gzip -d bigann_query.bvecs.gz
tar -xvf bigann_gnd.tar.gz

# This is very slow; consider tmux it
nohup wget -c ftp://ftp.irisa.fr/local/texmex/corpus/bigann_base.bvecs.gz > download_base.log 2>&1 &
