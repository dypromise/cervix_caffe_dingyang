#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=examples/imagenet
DATA=/mnt/lustre/cervix_caffe_dingyang/myself/data_resized
TOOLS=/mnt/lustre/cervix_caffe_dingyang/caffe/build/tools

$TOOLS/compute_image_mean cervix_train_leveldb \
	cervix_mean_binaryproto

echo "Done."
