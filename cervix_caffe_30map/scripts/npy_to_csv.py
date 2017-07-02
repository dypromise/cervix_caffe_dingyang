import os
import csv
import numpy as np
import copy
import sys
myself='/mnt/lustre/dingyang/cervix_caffe_dingyang/myself'
csvfile=myself+'/bvlc_googlenetBN/extracted_features/features.csv'
def to_csv(npys_dir,txt_dir):
    npyslist = os.listdir(npys_dir)
    print npyslist
    arr=np.zeros((512,3),dtype='float64')
    nums = len(npyslist)
    for npy_name in npyslist:
        arr=arr+ np.load(os.path.join(npys_dir,npy_name))
    features = arr/float(nums)

    csv_file=open(csvfile,'w')
    label_file = open(txt_dir,'r')
    writer=csv.writer(csv_file)
    writer.writerow(['image_name','Type_1','Type_2','Type_3'])
    n=features.shape[0]  # num of rows.
    for i in range(n):
        line = label_file.readline()
        image_name=line.split(' ')[0]
        data=([image_name,'%.16f'%features[i][0],'%.16f'%features[i][1],'%.16f'%features[i][2]])
        #image_num=int(image_name.split('.')[0])
        #  arr[image_num]= arr[image_num]+features[i]
    #  arr=arr/float(nums)
    #  for i in range(512):
        #  image_name = str(i)+'.jpg'
        #data=([image_name,'%.16f'%(arr[i][0]),'%.16f'%(arr[i][1]),'%.16f'%(arr[i][2])])
        writer.writerow(data)
    csv_file.close()
    label_file.close()

to_csv(myself+'/bvlc_googlenetBN/extracted_features/npys/',myself+'/test_lmdb_txt/test_resized.txt')
        
