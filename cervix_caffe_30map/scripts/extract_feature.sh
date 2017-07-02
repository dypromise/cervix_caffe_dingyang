n_gpus=1
cervix_dir=/mnt/lustre/dingyang/cervix_caffe_dingyang
caffe_dir=$cervix_dir/caffe
trained_model=/mnt/lustre/dingyang/cervix_caffe_dingyang/myself/bvlc_googlenetBN/caffe_bvlc_googlenetBNx4___iter_3000.caffemodel
model=$cervix_dir/myself/bvlc_googlenetBN/deploy3.prototxt
blob=prob
result_dir='/mnt/lustre/dingyang/cervix_caffe_dingyang/myself/bvlc_googlenetBN/extracted_features/'
num_iters=512
cd $result_dir
rm -rf extracted*
MV2_USE_CUDA=1 MV2_ENABLE_AFFINITY=0 MV2_SMP_USE_CMA=0 \
	srun -p AutoPilot --mpi=pmi2  --gres=gpu:$n_gpus -n$n_gpus --ntasks-per-node=$n_gpus \
    $caffe_dir/build/tools/extract_features \
    ${trained_model} ${model} ${blob},prob \
    ${result_dir}/extracted_features_lmdb,${result_dir}/extracted_labels_lmdb \
    ${num_iters} lmdb GPU 
echo "Converting LMDB formated features to npy format..."
cd npys
python /mnt/lustre/dingyang/cervix_caffe_dingyang/myself/bvlc_googlenetBN/scripts/convert_lmdb_to_numpy.py ../extracted_labels_lmdb features2.npy

#echo "Converting features of npy format to sorted CSV file in ../extracted_features."
#python $model_dir/scripts/npy_to_csv.py 
echo "Done."
