#!/usr/bin/env sh
set -e
n_gpus=4
cervix_dir=/mnt/lustre/cervix_caffe_dingyang
myself=$cervix_dir/myself
LOG=$myself/vggnet/logs/log-fixed.log
MV2_USE_CUDA=1 MV2_ENABLE_AFFINITY=0 MV2_SMP_USE_CMA=0 \
	srun -p Retrieval --mpi=pmi2 --gres=gpu:$n_gpus -n$n_gpus --ntasks-per-node=$n_gpus \
	$cervix_dir/caffe/build/tools/caffe train \
	--solver=$myself/vggnet/solver.prototxt \
	--gpu all \
	--weights=$myself/vggnet/VGG_ILSVRC_16_layers.caffemodel 2>&1 |tee $LOG 
