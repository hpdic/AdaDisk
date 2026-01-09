```bash
# Download data sets
git clone https://github.com/harsha-simhadri/big-ann-benchmarks.git
cd big-ann-benchmarks
source ~/hpdic/AdaDisk/venv/bin/activate
pip install azure-storage-blob
python3 create_dataset.py --dataset msspacev-1B
cd ~/hpdic/big-ann-benchmarks/data/MSSPACEV1B/
wget -c https://comp21storage.z5.web.core.windows.net/comp21/spacev1b/spacev1b_base.i8bin

# Prepare data sets
mkdir -p ~/hpdic/spacev1b_data
cd ~/hpdic/spacev1b_data
ln -s ~/hpdic/big-ann-benchmarks/data/MSSPACEV1B/spacev1b_base.i8bin spacev1b_base.bin
ln -s ~/hpdic/big-ann-benchmarks/data/MSSPACEV1B/query.i8bin spacev1b_query.bin
ln -s ~/hpdic/big-ann-benchmarks/data/MSSPACEV1B/public_query_gt100.bin spacev1b_ground_truth.bin

# Build baseline index
tmux
source ~/hpdic/AdaDisk/venv/bin/activate
cd ~/hpdic/AdaDisk/experiments/spacev1b
chmod +x run_build_baseline.sh
./run_build_baseline.sh 2>&1 | tee build_spacev1b.log
# ctrl+b d
cd ~/hpdic/AdaDisk/experiments/spacev1b
tail -f build_spacev1b.log

# Baseline query
cd ~/hpdic/AdaDisk/experiments/spacev1b
chmod +x run_query_baseline.sh
./run_query_baseline.sh

# Build MCGI index
pip install faiss-cpu numpy
python calculate_lid.py
tmux a
chmod +x run_build_mcgi.sh
./run_build_mcgi.sh 2>&1 | tee build_mcgi_spacev1b.log
# ctrl+b d
cd ~/hpdic/AdaDisk/experiments/spacev1b
tail -f build_mcgi_spacev1b.log

# MCGI query
cd ~/hpdic/AdaDisk/experiments/spacev1b
chmod +x run_search_mcgi.sh
./run_search_mcgi.sh
```