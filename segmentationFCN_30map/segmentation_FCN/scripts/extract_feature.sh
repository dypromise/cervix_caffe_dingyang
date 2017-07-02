n_gpus=1
cervix_dir=/mnt/lustre/dingyang/cervix_caffe_dingyang
caffe_dir=$cervix_dir/sensenet
model_dir=$cervix_dir/myself/segmentation_FCN
trained_model='/mnt/lustre/dingyang/cervix_caffe_dingyang/myself/segmentation_FCN/FCN_try_adloss_iter_16600.caffemodel'
model=$cervix_dir/myself/segmentation_FCN/voc-fcn32s-deploy.prototxt

result_dir=/mnt/lustre/dingyang
num_iters=34
rm -rf ${result_dir}/extracted_labels_lmdb

MV2_USE_CUDA=1 MV2_ENABLE_AFFINITY=0 MV2_SMP_USE_CMA=0 \
	srun -p Retrieval --mpi=pmi2 --gres=gpu:$n_gpus -n $n_gpus --ntasks-per-node=$n_gpus \
    $caffe_dir/example/build/tools/extract_features \
    ${trained_model} ${model} mask_dy \
    ${result_dir}/extracted_labels_lmdb \
    ${num_iters} lmdb GPU

echo "Converting LMDB formated features to npy format..."
python $model_dir/scripts/convert_lmdb_to_numpy.py ${result_dir}/extracted_labels_lmdb /mnt/lustre/dingyang/f.npy
echo "Done."