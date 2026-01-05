# If you have at least 600 GB of free disk space, try the one billion
# vectors dataset from the BigANN benchmark:
```bash
mkdir -p ~/hpdic/sift1b_data
cd ~/hpdic/sift1b_data

wget -c ftp://ftp.irisa.fr/local/texmex/corpus/bigann_query.bvecs.gz
wget -c ftp://ftp.irisa.fr/local/texmex/corpus/bigann_gnd.tar.gz
gzip -d bigann_query.bvecs.gz
tar -xvf bigann_gnd.tar.gz

# The following will be very slow; consider scping it from another machine
nohup wget -c ftp://ftp.irisa.fr/local/texmex/corpus/bigann_base.bvecs.gz > download_base.log 2>&1 &
gzip -d bigann_base.bvecs.gz
cp ~/hpdic/AdaDisk/experiments/bigann/convert_sift1b.py ~/hpdic/sift1b_data/.
python3 convert_sift1b.py

# Baseline index building
tmux
chmod +x ~/hpdic/AdaDisk/experiments/bigann/run_build_baseline.sh
cd ~/hpdic/AdaDisk
# You should take a look at the parameters in the following script
./experiments/bigann/run_build_baseline.sh 2>&1 | tee build_sift1b.log
# ctrl+b d
cd ~/hpdic/AdaDisk
tail -f build_sift1b.log

# Baseline query
cd ~/hpdic/sift1b_data
source ../AdaDisk/venv/bin/activate
python ../AdaDisk/experiments/bigann/convert_search.py
sudo sysctl -w fs.aio-max-nr=1048576
bash ../AdaDisk/experiments/bigann/run_search_baseline.sh

# MCGI index building
cd ~/hpdic/AdaDisk
tmux
chmod +x ~/hpdic/AdaDisk/experiments/bigann/run_build_mcgi.sh
# You should take a look at the parameters in the following script
./experiments/bigann/run_build_mcgi.sh 2>&1 | tee build_mcgi.log
# ctrl+b d
cd ~/hpdic/AdaDisk
tail -f build_mcgi.log

# MCGI query
cd ~/hpdic/sift1b_data
source ../AdaDisk/venv/bin/activate
python ../AdaDisk/experiments/bigann/convert_search.py
sudo sysctl -w fs.aio-max-nr=1048576
bash ../AdaDisk/experiments/bigann/run_search_mcgi.sh
```