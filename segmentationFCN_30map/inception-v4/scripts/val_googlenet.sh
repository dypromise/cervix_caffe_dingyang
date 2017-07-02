#!/usr/bin/env sh
set -e
n_gpus=4
cervix_dir=/mnt/lustre/dingyang/cervix_caffe_dingyang
caffe_dir=$cervix_dir/caffe
model_dir=$cervix_dir/myself/bvlc_googlenet
LOG=$model_dir/logs/test_5-13.log
MV2_USE_CUDA=1 MV2_ENABLE_AFFINITY=0 MV2_SMP_USE_CMA=0 \
	srun -p Retrieval --mpi=pmi2 --gres=gpu:$n_gpus -n$n_gpus --ntasks-per-node=$n_gpus \
	$caffe_dir/build/tools/caffe test \
	--gpu all \
	--model=$model_dir/train_val.prototxt \
	--weights=$model_dir/bvlc_googlenet_iter_50000.caffemodel 2>&1 |tee $LOG
