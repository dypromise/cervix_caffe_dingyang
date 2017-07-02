#!/usr/bin/env sh
set -e
n_gpus=8
cervix_dir=/mnt/lustre/dingyang/cervix_caffe_dingyang
caffe_dir=/mnt/lustre/dingyang/cervix_caffe_dingyang/sensenet
model_dir=$cervix_dir/myself/inception-v4
LOG=$model_dir/logs/inceptionv4_tryZXYBN_ADLossb1b7_FCN_RR.log
MV2_USE_CUDA=1 MV2_ENABLE_AFFINITY=0 MV2_SMP_USE_CMA=0 \
	srun -p ShareXP --mpi=pmi2 --gres=gpu:$n_gpus -n $n_gpus --ntasks-per-node=$n_gpus \
	$caffe_dir/example/build/tools/caffe train \
	--solver=$model_dir/solver_v4_ZXYBN_ADLossb1b7_FCN.prototxt \
	--weights=/mnt/lustre/dingyang/cervix_caffe_dingyang/myself/segmentation_FCN/savedCaffemodels/FCN_try_adloss_iter_16600.caffemodel,/mnt/lustre/dingyang/cervix_caffe_dingyang/myself/inception-v4/inception_v4.caffemodel \
   	2>&1 |tee $LOG
