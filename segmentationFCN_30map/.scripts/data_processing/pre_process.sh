root='/mnt/lustre/dingyang/TO_DINGYANG'
origin_train_dir="${root}/origin_train"
additional_data_dir="${root}/additional_data"

# cd ${root}
# mkdir train_new
# mkdir val_new
# cd train_new
# echo "copying."
# for type in Type_1 Type_2 Type_3
# do
    # mkdir "$type"
    # echo "$type"
    # cp ${origin_train_dir}/${type}/*.jpg ./$type/
    # cp ${additional_data_dir}/${type}/*.jpg ./$type/
#done

echo  "seperating train and val dir"
for type in Type_1 Type_2 Type_3
do
	if [ "$type" == "Type_1" ]; then 
		VALI_NUM=300
	elif [ "$type" == "Type_2" ]; then
		VALI_NUM=800
	else
		VALI_NUM=500
	fi

    val_dir=${root}/val_new
    train_dir=${root}/train_new
    # Move the first randomly selected validation_ratio*train_samples images to the validation set.
    val_imgs=$(ls -1 ${train_dir}/$type | shuf | head -$VALI_NUM)
	for img in ${val_imgs}; do
		mv -f ${train_dir}/${type}/${img} ${val_dir}
	done
done
