#!/usr/bin/env sh
# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs
set -e

DATA=/mnt/lustre/dingyang/TO_DINGYANG/resized_test/
TXT=/mnt/lustre/dingyang/TO_DINGYANG/test.txt
TOOLS=/mnt/lustre/dingyang/cervix_caffe_dingyang/caffe/build/tools

# already been resized using another tool.
RESIZE=false
if $RESIZE; then
  RESIZE_HEIGHT=256
  RESIZE_WIDTH=256
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

echo "Creating test lmdb..."
GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    $DATA \
    $TXT \
    test_lmdb 1

echo "Done."
