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
def label_test(test_dir,txtname):
    files = os.listdir(test_dir)
    file_write = open(os.path.join(test_dir,txtname),'w')
    for image in files:
        if(image[-3:]=='jpg'):
            file_write.write(image+' '+'0'+'\n')
    file_write.close()


def label_train_FCN(train_dir,txtname):
    types = os.listdir(train_dir)
    print(types)
    file_txt = open(os.path.join(train_dir,txtname),'w')
    for type in types:
        if(type[-6:-2]=='Type'):
            type_num = int(type[-1])-1
            filenames = os.listdir(os.path.join(train_dir,type))
            for jpg_file in filenames:
                if(jpg_file[-3:]=='jpg'):
                    file_txt.write(type+'/'+jpg_file+' '+'labels'+'/'+jpg_file+'_label.png'+'\n')
    file_txt.close()

def label_val_FCN(val_dir,txtname):
    files = os.listdir(val_dir)
    file_write = open(os.path.join(val_dir,txtname),'w')
    for image in files:
        if(image[-3:]=='jpg'):
            type=int(image.split('_')[2])-1
            file_write.write(image+' '+'labels/'+image+'_label.png'+'\n')
    file_write.close()
def label_test_FCN(test_dir,txtname):
    files = os.listdir(test_dir)
    file_write = open(os.path.join(test_dir,txtname),'w')
    for image in files:
        if(image[-3:]=='jpg'):
            file_write.write(image+' '+'0'+'\n')
    file_write.close()    

# label_train_FCN('/mnt/lustre/dingyang/train_FCNx4','train_FCNx4.txt')
# label_val_FCN('/mnt/lustre/dingyang/val_FCNx4','val_FCNx4.txt')
label_test('/mnt/lustre/dingyang/test_stg2_375/test0','test0.txt')
label_test('/mnt/lustre/dingyang/test_stg2_375/test1','test1.txt')
label_test('/mnt/lustre/dingyang/test_stg2_375/test3','test3.txt')
label_test('/mnt/lustre/dingyang/test_stg2_375/test4','test4.txt')
label_test('/mnt/lustre/dingyang/test_stg2_375/test5','test5.txt')
label_test('/mnt/lustre/dingyang/test_stg2_375/test6','test6.txt')
label_test('/mnt/lustre/dingyang/test_stg2_375/test2','test2.txt')

# label_train('/mnt/lustre/dingyang/train_rescale_rotation')
#label_val('/mnt/lustre/dingyang/TO_DINGYANG/val_fr')
# label_test('/mnt/lustre/dingyang/TO_DINGYANG/test_fr')

