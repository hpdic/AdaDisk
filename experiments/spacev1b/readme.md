```bash
git clone https://github.com/harsha-simhadri/big-ann-benchmarks.git
cd big-ann-benchmarks
source ~/hpdic/AdaDisk/venv/bin/activate
pip install azure-storage-blob
python3 create_dataset.py --dataset msspacev-1B
cd ~/hpdic/big-ann-benchmarks/data/MSSPACEV1B/
wget -c https://comp21storage.z5.web.core.windows.net/comp21/spacev1b/spacev1b_base.i8bin
```