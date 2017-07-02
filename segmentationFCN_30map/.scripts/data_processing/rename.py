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


