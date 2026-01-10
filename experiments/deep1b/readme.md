```bash
# Download data
python download_deep1b_10m.py
wget -O deep1b_query.fbin http://ann-benchmarks.com/deep-image-96-angular.hdf5
# Generate groundtruth
python gen_query.py
cd ~/hpdic/AdaDisk/build/apps/utils
./compute_groundtruth \
  --data_type float \
  --dist_fn l2 \
  --base_file /home/cc/hpdic/deep1b_data/deep1b_base_1M.fbin \
  --query_file /home/cc/hpdic/deep1b_data/deep1b_query.fbin \
  --gt_file /home/cc/hpdic/deep1b_data/deep1b_10M_gt.bin \
  --K 100
```