import cv2
import os


def resize_images_and_output(origin_dir,output_dir):
    files = os.listdir(origin_dir)
    count=0
    for image in files:
        count+=1
        if(image[-3:]=='jpg'):
            try:
                print(image)
                print(count)     
                img = cv2.imread(os.path.join(origin_dir,image))
                dst = cv2.resize(img,(256,256))
                cv2.imwrite(os.path.join(output_dir,image),dst)
            except:
                print("cathc EXEPTION")
                continue

resize_images_and_output('/mnt/lustre/dingyang/TO_DINGYANG/train_new/Type_3','/mnt/lustre/dingyang/TO_DINGYANG/reized_data/train/Type_3')
#  train_dir = '/mnt/lustre/dingyang/TO_DINGYANG/data_dir/train'
#  val_dir='/mnt/lustre/dingyang/TO_DINGYANG/data_dir/validation/validation_new'
#  train_out_dir='/mnt/lustre/dingyang/TO_DINGYANG/data_dir/data_resized/train'
#  val_out_dir= '/mnt/lustre/dingyang/TO_DINGYANG/data_dir/data_resized/validation'

#for typefile in os.listdir(train_dir):
    #  if(typefile[0:4]=='Type'):
        #  resize_images_and_output(os.path.join(train_dir,typefile),os.path.join(train_out_dir,typefile))
