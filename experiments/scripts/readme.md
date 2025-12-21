```bash
cd ~/DiskANN

# download glove (0.48 GB, 100 Dim)
python3 experiments/scripts/get_glove.py

# download SIFT and GIST
python3 experiments/scripts/get_data.py

# calculate LID
OPENBLAS_NUM_THREADS=32 python3 experiments/scripts/calc_lid.py experiments/data/glove/glove_base.bin
OPENBLAS_NUM_THREADS=8 python3 experiments/scripts/calc_lid.py experiments/data/sift/sift_base.bin
OPENBLAS_NUM_THREADS=8 python3 experiments/scripts/calc_lid.py experiments/data/gist/gist_base.bin

# run GloVe
OPENBLAS_NUM_THREADS=32 bash experiments/scripts/run_exp.sh glove

# run SIFT
bash experiments/scripts/run_exp.sh sift

# run GIST
bash experiments/scripts/run_exp.sh gist
```