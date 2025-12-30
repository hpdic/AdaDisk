```bash
cd ~/AdaDisk

# download glove (0.48 GB, 100 Dim)
python3 experiments/scripts/get_glove.py

# download SIFT and GIST
python3 experiments/scripts/get_data.py

# calculate LID
OPENBLAS_NUM_THREADS=32 python3 experiments/scripts/calc_lid.py experiments/data/glove/glove_base.bin
OPENBLAS_NUM_THREADS=32 python3 experiments/scripts/calc_lid.py experiments/data/sift/sift_base.bin
OPENBLAS_NUM_THREADS=32 python3 experiments/scripts/calc_lid.py experiments/data/gist/gist_base.bin

# analysize LID
python3 experiments/scripts/analyze_lid.py

# run GloVe
OPENBLAS_NUM_THREADS=32 bash experiments/scripts/run_exp.sh glove

# run SIFT
OPENBLAS_NUM_THREADS=32 bash experiments/scripts/run_exp.sh sift

# run GIST
OPENBLAS_NUM_THREADS=32 bash experiments/scripts/run_exp.sh gist

# run full scan
OPENBLAS_NUM_THREADS=32 bash experiments/scripts/full_scan.sh

# more specific scan
OPENBLAS_NUM_THREADS=32 bash experiments/scripts/scan_patch.sh

# Faiss baseline
pip install faiss-cpu numpy
```