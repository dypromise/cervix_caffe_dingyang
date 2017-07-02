import cv2
import os
import numpy as np
import math

def resize_images_and_output(origin_dir,output_dir):
    files = os.listdir(origin_dir)
    count=0
    for image in files:
        count+=1
        if(image[-3:]=='jpg'):
            try:
                print(image)
                print(count)
                img = cv2.imread(os.path.join(origin_dir, image))
                aa=img.shape
               
                #img = cv2.resize(img, dsize=tile_size)
                img = cv2.resize(img, (375,375))
                cv2.imwrite(os.path.join(output_dir, image), img)
                for i in range(3):
                    #  w = img.shape[1]
                    #  h = img.shape[0]
                    #  angle=90
                    #  rangle = np.deg2rad(angle)  # angle in radians
                    #  scale=1.0
                    #  # now calculate new image width and height
                    #  nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
                    #  nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
                    #  rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
                    #  rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
                    #  rot_mat[0, 2] += rot_move[0]
                    #  rot_mat[1, 2] += rot_move[1]
                    #  img=cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)
                    #  cv2.imwrite(os.path.join(output_dir, image[:-4] + '_' + str(i + 1) +'.jpg'), img)
                    rows, cols = img.shape[:2]
                    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), (i+1)*90, 1)
                    res = cv2.warpAffine(img, M, (rows, cols))
                    cv2.imwrite(os.path.join(output_dir, image[:-4] + '_' + str(i + 1) + '.jpg'), res)
            except:
                print("catch except.")
                continue
def rotation_x4(origin_dir,output_dir):
    files = os.listdir(origin_dir)
    count=0
    for image in files:
        count+=1
        if(image[-3:]=='png'):
            try:
                print(image)
                print(count)
                img = cv2.imread(os.path.join(origin_dir, image))
                aa=img.shape
                # if (img.shape[0] > img.shape[1]):
                #     tile_size = (256,int(img.shape[0] * 256 / img.shape[1]))
                # else:
                #     tile_size = (int(img.shape[1] * 256 / img.shape[0]),256)
                # #img = cv2.resize(img, dsize=tile_size)
                # img = cv2.resize(img, (256,256))
                cv2.imwrite(os.path.join(output_dir, image), img)
                for i in range(3):
                    #  w = img.shape[1]
                    #  h = img.shape[0]
                    #  angle=90
                    #  rangle = np.deg2rad(angle)  # angle in radians
                    #  scale=1.0
                    #  # now calculate new image width and height
                    #  nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
                    #  nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
                    #  rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
                    #  rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
                    #  rot_mat[0, 2] += rot_move[0]
                    #  rot_mat[1, 2] += rot_move[1]
                    #  img=cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)
                    #  cv2.imwrite(os.path.join(output_dir, image[:-4] + '_' + str(i + 1) +'.jpg'), img)
                    rows, cols = img.shape[:2]
                    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), (i+1)*90, 1)
                    res = cv2.warpAffine(img, M, (rows, cols))
                    cv2.imwrite(os.path.join(output_dir, image[:-14] + '_' + str(i + 1) + '.jpg_label.png'), res)
            except:
                print("catch except.")
                continue


def rescale(origin_dir, output_dir, image_size):
    files = os.listdir(origin_dir)
    count = 0
    for image in files:
        count += 1
        if (image[-3:] == 'jpg'):
            # try:
            print(image)
            print(count)
            img = cv2.imread(os.path.join(origin_dir, image))

            img_1 = cv2.resize(img, (468, 468))
            cv2.imwrite(os.path.join(output_dir, image[:-4] + '_scale_' + str(1) + '.jpg'), img_1)
            img_2 = cv2.resize(img, (562, 562))
            cv2.imwrite(os.path.join(output_dir, image[:-4] + '_scale_' + str(2) + '.jpg'), img_2)
            img_3 = cv2.resize(img, (675, 675))
            cv2.imwrite(os.path.join(output_dir, image[:-4] + '_scale_' + str(3) + '.jpg'), img_3)

resize_images_and_output('/mnt/lustre/dingyang/test_stg2/test_stg2_6', '/mnt/lustre/dingyang/test_stg2_375/test6')
#rescale('/mnt/lustre/dingyang/train_','/mnt/lustre/dingyang/test_stg2/test6')
#  train_dir = '/mnt/lustre/dingyang/TO_DINGYANG/data_dir/train'
#  val_dir='/mnt/lustre/dingyang/TO_DINGYANG/data_dir/validation/validation_new'
#  train_out_dir='/mnt/lustre/dingyang/TO_DINGYANG/data_dir/data_resized/train'
#  val_out_dir= '/mnt/lustre/dingyang/TO_DINGYANG/data_dir/data_resized/validation'

#for typefile in os.listdir(train_dir):
    #  if(typefile[0:4]=='Type'):
        #  resize_images_and_output(os.path.join(train_dir,typefile),os.path.join(train_out_dir,typefile))
