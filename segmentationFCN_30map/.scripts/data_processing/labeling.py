import sys
import os

def label_train(train_dir):
    types = os.listdir(train_dir)
    print(types)
    file_txt = open(os.path.join(train_dir,'train.txt'),'w')
    for type in types:
        if(type[0:4]=='Type'):
            type_num = int(type[-1])-1
            filenames = os.listdir(os.path.join(train_dir,type))
            for jpg_file in filenames:
                if(jpg_file[-3:]=='jpg'):
                    file_txt.write(type+'/'+jpg_file+' '+str(type_num)+'\n')
    file_txt.close()

def label_val(val_dir):
    files = os.listdir(val_dir)
    file_write = open(os.path.join(val_dir,'val.txt'),'w')
    for image in files:
        if(image[-3:]=='jpg'):
            type=int(image.split('_')[2])-1
            file_write.write(image+' '+str(type)+'\n')
    file_write.close()

label_train('/mnt/lustre/dingyang/TO_DINGYANG/reized_data/train')
label_val('/mnt/lustre/dingyang/TO_DINGYANG/reized_data/val')


