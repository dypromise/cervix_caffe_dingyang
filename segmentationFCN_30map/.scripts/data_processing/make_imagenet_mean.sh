#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

TRAIN_PROTO=/mnt/lustre/cervix_caffe_dingyang/myself/leveldb_and_mean/train_lmdb
TOOLS=/mnt/lustre/cervix_caffe_dingyang/caffe/build/tools
OUTPUT=/mnt/lustre/cervix_caffe_dingyang/myself/leveldb_and_mean


$TOOLS/compute_image_mean $TRAIN_PROTO $OUTPUT/mean.binaryproto
echo "Done."
