#!/usr/bin/env sh
set -e
partation=Test
model_def=/mnt/lustre/dingyang/cervix_caffe_dingyang/myself/inception-v4/inceptionV4_deploy_ADLoss_b1b7.prototxt
model_weights=/mnt/lustre/dingyang/cervix_caffe_dingyang/myself/inception-v4/ensambles_model/rescale_rotation_57000jie_iter_5200.caffemodel
test_root=/mnt/lustre/dingyang/test_stg2
LOG=/mnt/lustre/dingyang/cccc.log
# srun -p $partation --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 python classify_batch.py $model_def $model_weights $test_root/test0 $test_root/test0.txt 0 2>&1 |tee $LOG &
# srun -p $partation --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 python classify_batch.py $model_def $model_weights $test_root/test1 $test_root/test1.txt 1 & 
# srun -p $partation --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 python classify_batch.py $model_def $model_weights $test_root/test2 $test_root/test2.txt 2 & 
# srun -p $partation --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 python classify_batch.py $model_def $model_weights $test_root/test3 $test_root/test3.txt 3 & 
srun -p $partation --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 python classify_batch.py $model_def $model_weights $test_root/test4 $test_root/test4.txt 4 & 
srun -p $partation --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 python classify_batch.py $model_def $model_weights $test_root/test5 $test_root/test5.txt 5 & 
srun -p $partation --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 python classify_batch.py $model_def $model_weights $test_root/test6 $test_root/test6.txt 6 & 
srun -p $partation --mpi=pmi2 --gres=gpu:1 -n 1 --ntasks-per-node=1 python classify_batch.py $model_def $model_weights $test_root/test7 $test_root/test7.txt 7 & 
