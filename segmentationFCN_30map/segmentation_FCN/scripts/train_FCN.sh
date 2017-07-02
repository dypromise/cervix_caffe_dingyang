#!/usr/bin/env sh
set -e
n_gpus=4
cervix_dir=/mnt/lustre/dingyang/cervix_caffe_dingyang
caffe_dir=/mnt/lustre/dingyang/cervix_caffe_dingyang/sensenet-release
model_dir=$cervix_dir/myself/segmentation_FCN
LOG=$model_dir/logs/FCN_try_adloss.log
MV2_USE_CUDA=1 MV2_ENABLE_AFFINITY=0 MV2_SMP_USE_CMA=0 \
	srun -p ShareXP --mpi=pmi2 --gres=gpu:$n_gpus -n $n_gpus --ntasks-per-node=$n_gpus \
	$caffe_dir/example/build/tools/caffe train \
	--solver=$model_dir/solver.prototxt \
	--weights=/mnt/lustre/dingyang/cervix_caffe_dingyang/myself/segmentation_FCN/fcn32s-heavy-pascal.caffemodel \
   	2>&1 |tee $LOG
