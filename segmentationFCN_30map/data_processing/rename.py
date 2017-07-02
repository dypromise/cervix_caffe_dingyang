import os
import sys
def rename(file_dir,prefix):
    types = os.listdir(file_dir)
    for type in types:
        if(type[0:4]=='Type'):
            imagefiles = os.listdir(os.path.join(file_dir,type))
            for image in imagefiles:
                if(image[-3:]=='jpg'):
                    newname = prefix+'_'+type+'_'+image
                    os.rename(os.path.join(file_dir,type,image),os.path.join(file_dir,type,newname))
#process('/mnt/lustre/dingyang/TO_DINGYANG/data_dir/validation')



def rename_test(file_dir):
    fileslist=os.listdir(file_dir)
    for file in fileslist:
        if(file[-3:]=='jpg'):
            num=file.split('.')[0]
            num=num.zfill(3)
            newname=num+'.jpg'
            os.rename(os.path.join(file_dir,file),os.path.join(file_dir,newname))
#rename_test('/mnt/lustre/dingyang/TO_DINGYANG/resized_test')
def unrename_test(file_dir):
    fileslist=os.listdir(file_dir)
    for file in fileslist:
        if(file[-3:]=='jpg'):
            num=file.split('.')[0]
            num=int(num)
            newname=str(num)+'.jpg'
            os.rename(os.path.join(file_dir,file),os.path.join(file_dir,newname))
unrename_test('/mnt/lustre/dingyang/TO_DINGYANG/resized_test')
