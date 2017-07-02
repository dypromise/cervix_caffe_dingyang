#!/usr/bin/env sh
set -e
n_gpus=2
LOG=/mnt/lustre/cervix_caffe_dingyang/myself/scripts/googlenet1/log-fixed_b128.log
MV2_USE_CUDA=1 MV2_ENABLE_AFFINITY=0 MV2_SMP_USE_CMA=0 \
	srun -p Retrieval --mpi=pmi2 --gres=gpu:$n_gpus -n$n_gpus --ntasks-per-node=$n_gpus \
	/mnt/lustre/cervix_caffe_dingyang/caffe/build/tools/caffe train \
	--solver=/mnt/lustre/cervix_caffe_dingyang/myself/bvlc_googlenet/solver.prototxt \
	--gpu all \
	--weights=/mnt/lustre/cervix_caffe_dingyang/myself/bvlc_googlenet/bvlc_googlenet.caffemodel 2>&1 |tee $LOG
