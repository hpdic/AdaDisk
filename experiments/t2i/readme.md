```bash
source ~/hpdic/AdaDisk/venv/bin/activate
python download_t2i_hf.py 

cd ~/hpdic/AdaDisk/build/apps/utils

./compute_groundtruth \
  --data_type float \
  --dist_fn l2 \
  --base_file /home/cc/hpdic/t2i_data/t2i_base_1M.fbin \
  --query_file /home/cc/hpdic/t2i_data/t2i_query.fbin \
  --gt_file /home/cc/hpdic/t2i_data/t2i_1M_gt.bin \
  --K 100

```