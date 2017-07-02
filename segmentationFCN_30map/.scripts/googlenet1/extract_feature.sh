n_gpus=1
# LOG=/mnt/lustre/cervix_caffe_dingyang/myself/bvlc_googlenet/logs/log-'data +%Y-%m-%d-%H-%S'.log
cervix_dir=/mnt/lustre/cervix_caffe_dingyang
caffe_dir=$cervix_dir/caffe
trained_model=$cervix_dir/myself/bvlc_googlenet/bvlc_googlenet_iter_20000.caffemodel
model=$cervix_dir/myself/bvlc_googlenet/deploy_2.prototxt
blob=prob
result_dir=/mnt/lustre/dingyang/extracted_features_caffe
num_iters=33
subset=ff
cd $result_dir
rm -rf ff*
MV2_USE_CUDA=1 MV2_ENABLE_AFFINITY=0 MV2_SMP_USE_CMA=0 \
	srun -p AutoPilot --mpi=pmi2 --gres=gpu:$n_gpus -n$n_gpus --ntasks-per-node=$n_gpus \
    $caffe_dir/build/tools/extract_features \
    ${trained_model} ${model} ${blob},prob \
    ${result_dir}/${subset}_features_lmdb,${result_dir}/${subset}_labels_lmdb \
    ${num_iters} lmdb GPU

python convert_lmdb_to_numpy.py ff_labels_lmdb ../features.npy
