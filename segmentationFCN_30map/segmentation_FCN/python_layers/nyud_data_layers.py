import caffe

import numpy as np
import cv2
import random
import os.path as osp

class NYUDDataLayer(caffe.Layer):

    def setup(self, bottom, top):
        # config
        params = eval(self.param_str)
        self.source = params['source']
        self.mean = np.array(params['mean'])
        self.shuffle = params.get('shuffle', False)
        self.root_dir = params.get('root_dir', '')
        self.batch_size = params.get('batch_size', 1)
        self.height = params.get('height')
        self.width = params.get('width')
        self.mirror = params.get('mirror', False)
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        self.idx = 0
        
        self.images_list = self.load_images_list()
        self.images_num = len(self.images_list)

        if self.shuffle:
            random.shuffle(self.images_list)

        self.images_list = self.images_list + self.images_list[:self.batch_size]

    def reshape(self, bottom, top):
        top[0].reshape(self.batch_size, 3, self.height, self.width)
        top[1].reshape(self.batch_size, 1, self.height, self.width)

    def forward(self, bottom, top):
        image, label = self.load_batch()
        if self.mirror:
            for i in xrange(self.batch_size):
                if random.randint(0,1) == 0:
                    image[i] = image[i, :, :, ::-1]
                    label[i] = label[i, :, :, ::-1]
        # assign output
        top[0].data[...] = image
        top[1].data[...] = label

    def backward(self, top, propagate_down, bottom):
        pass

    def load_images_list(self):
        lines = open(self.source, 'r').readlines()
        images_list = []
        for line in lines:
            image_path, label_path = line.split()
            #if not osp.exists(osp.join(self.root_dir, image_path)):
            #    print 'Error!!! Image not exists!!!'
            #    print osp.join(self.root_dir, image_path)
            #if not osp.exists(osp.join(self.root_dir, label_path)):
            #    print 'Error!!! Label not exists!!!'
            #    print osp.join(self.root_dir, label_path)
            images_list.append([image_path, label_path])
        return images_list

    def read_image(self,image_info):
        image = cv2.imread(osp.join(self.root_dir, image_info[0])).astype('float64')
        image = cv2.resize(image, (self.height, self.width))
        image = image.transpose(2, 0, 1)
        image[0,:,:] -= self.mean[0]
        image[1,:,:] -= self.mean[1]
        image[2,:,:] -= self.mean[2]
        label = cv2.imread(osp.join(self.root_dir, image_info[1]),0)
        label = cv2.resize(label, (self.height, self.width))
        _, thresh_binary = cv2.threshold(label , 10 , 1 , cv2.THRESH_BINARY)
        label = thresh_binary.reshape((1,self.height,self.width))
        return np.concatenate((image, label), axis=0) 

    def load_batch(self):
        batch_data = np.array(map(self.read_image, 
                self.images_list[self.idx:self.idx+self.batch_size]))
        if self.idx + self.batch_size < self.images_num:
            self.idx += self.batch_size
        else:    
            if self.shuffle:
                self.idx = 0
                tmp = self.images_list[:self.images_num]
                random.shuffle(tmp)
                self.images_list = tmp + tmp[:self.batch_size]
            else:
                self.idx = self.idx + self.batch_size - self.images_num
        return batch_data[:,:3,:,:], batch_data[:,3:4,:,:]
