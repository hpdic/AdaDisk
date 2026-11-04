#!/bin/bash
# Real-search check of the query-adaptive L: stock vs MCGI_ADAPT on the odd queries.
# Usage: adaptL_validate.sh out_dir data_type index_prefix query_odd gt_odd "L list" "MCGI_ADAPT" [extra args]
out=$1; dt=$2; idx=$3; q=$4; gt=$5; Ls=$6; ad=$7; shift 7
S=/home/cc/hpdic/adadisk/build/apps/search_disk_index
mkdir -p "$out" && cd "$out" || exit 1
export TMPDIR=$HOME/data/mcgi/tmp
ARGS="--data_type $dt --dist_fn l2 --index_path_prefix $idx --query_file $q --gt_file $gt -K 10 -L $Ls -T 32 $*"
$S $ARGS --result_path stock > stock.log 2>&1 &
a=$!
MCGI_ADAPT=$ad $S $ARGS --result_path adapt > adapt.log 2>&1 &
b=$!
wait $a $b
rm -f stock_* adapt_*
