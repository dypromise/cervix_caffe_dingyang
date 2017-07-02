#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

TRAIN_PROTO=/mnt/lustre/dingyang/cervix_caffe_dingyang/myself/bvlc_googlenet/leveldb_and_mean/fr_train_lmdb
TOOLS=/mnt/lustre/dingyang/cervix_caffe_dingyang/caffe/build/tools
OUTPUT=/mnt/lustre/dingyang/cervix_caffe_dingyang/myself/bvlc_googlenet/leveldb_and_mean


$TOOLS/compute_image_mean $TRAIN_PROTO $OUTPUT/fr_mean.binaryproto
echo "Done."
