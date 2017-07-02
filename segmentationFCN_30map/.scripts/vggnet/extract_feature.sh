n_gpus=1
caffe_dir=/mnt/lustre/cervix_caffe_dingyang/caffe
myself=/mnt/lustre/cervix_caffe_dingyang/myself
trained_model=$myself/bvlc_googlenet/bvlc_googlenet_iter_20000.caffemodel
model=$myself/bvlc_googlenet/deploy_2.prototxt
blob=conv1_1
result_dir=/mnt/lustre/dingyang
num_iters=161
subset=vggnet_16l
MV2_USE_CUDA=1 MV2_ENABLE_AFFINITY=0 MV2_SMP_USE_CMA=0 \
	srun -p AutoPilot --mpi=pmi2 --gres=gpu:$n_gpus -n$n_gpus --ntasks-per-node=$n_gpus \
    ${caffe_dir}/build/tools/extract_features \
    ${trained_model} ${model} ${blob},prob \
    ${result_dir}/${subset}_features_lmdb,${result_dir}/${subset}_labels_lmdb \
    ${num_iters} lmdb GPU
