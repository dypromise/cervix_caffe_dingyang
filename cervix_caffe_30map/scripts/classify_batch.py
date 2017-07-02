#!/usr/bin/env python
"""
Classifier is an image classifier specialization of Net.
"""
import numpy as np
# The caffe module needs to be on the Python path;
#  we'll add it here explicitly.
import sys
caffe_root = '/mnt/lustre/dingyang/cervix_caffe_dingyang/sensenet-release/'  
myself='/mnt/lustre/dingyang/cervix_caffe_dingyang/myself/'
sys.path.insert(0, caffe_root + 'core/python')
import caffe
caffe.mpi_init()
import os
import csv



class Classifier(caffe.Net):
    """
    Classifier extends Net for image class prediction
    by scaling, center cropping, or oversampling.
    Parameters
    ----------
    image_dims : dimensions to scale input for cropping/sampling.
        Default is to scale to net input size for whole-image crop.
    mean, input_scale, raw_scale, channel_swap: params for
        preprocessing options.
    """
    def __init__(self, model_file, pretrained_file, image_dims=None,
            mean=None, raw_scale=255,
            channel_swap=(2,1,0), input_scale=None):
        caffe.Net.__init__(self, model_file, pretrained_file, caffe.TEST)

        # configure pre-processing
        in_ = self.inputs[0]
        self.transformer = caffe.io.Transformer(
                {in_: self.blobs[in_].data.shape})
        self.transformer.set_transpose(in_, (2, 0, 1))
        if mean is not None:
            self.transformer.set_mean(in_, mean)
        if input_scale is not None:
            self.transformer.set_input_scale(in_, input_scale)
        if raw_scale is not None:
            self.transformer.set_raw_scale(in_, raw_scale)
        if channel_swap is not None:
            self.transformer.set_channel_swap(in_, channel_swap)

        self.crop_dims = np.array(self.blobs[in_].data.shape[2:])
        if not image_dims:
            image_dims = self.crop_dims
        self.image_dims = image_dims

    def predict(self, inputs, oversample=True):
        """
        Predict classification probabilities of inputs.
        Parameters
        ----------
        inputs : iterable of (H x W x K) input ndarrays.
        oversample : boolean
            average predictions across center, corners, and mirrors
            when True (default). Center-only prediction when False.
        Returns
        -------
        predictions: (N x C) ndarray of class probabilities for N images and C
            classes.
        """
        # Scale to standardize input dimensions.
        input_ = np.zeros((len(inputs),
            self.image_dims[0],
            self.image_dims[1],
            inputs[0].shape[2]),
            dtype=np.float32)
        for ix, in_ in enumerate(inputs):
            input_[ix] = caffe.io.resize_image(in_, self.image_dims)

        if oversample:
            # Generate center, corner, and mirrored crops.
            input_ = caffe.io.oversample(input_, self.crop_dims)
        else:
            # Take center crop.
            center = np.array(self.image_dims) / 2.0
            crop = np.tile(center, (1, 2))[0] + np.concatenate([
                -self.crop_dims / 2.0,
                self.crop_dims / 2.0
                ])
            crop = crop.astype(int)
            input_ = input_[:, crop[0]:crop[2], crop[1]:crop[3], :]

        # Classify
        caffe_in = np.zeros(np.array(input_.shape)[[0, 3, 1, 2]],
                dtype=np.float32)
        for ix, in_ in enumerate(input_):
            caffe_in[ix] = self.transformer.preprocess(self.inputs[0], in_)
        out = self.forward_all(**{self.inputs[0]: caffe_in})
        predictions_1 = out[self.outputs[0]]
        predictions_2 = out[self.outputs[1]]
        predictions_3 = out[self.outputs[2]]

        # For oversampling, average predictions across crops.
        if oversample:
            predictions_1 = predictions_1.reshape((len(predictions_1) / 10, 10, -1))
            predictions_2 = predictions_2.reshape((len(predictions_2) / 10, 10, -1))
            predictions_3 = predictions_3.reshape((len(predictions_3) / 10, 10, -1))
            predictions_1 = predictions_1.mean(1)
            predictions_2 = predictions_2.mean(1)
            predictions_3 = predictions_3.mean(1)
        return predictions_1,predictions_2,predictions_3




def make_inputs(image_dir,textfile):
    file_=open(textfile,'r')
    lines=file_.readlines()
    inputs=[]
    for line in lines:
        image=line.split(' ')[0]
        print(image)
        img_ = caffe.io.load_image(os.path.join(image_dir,image))
        inputs.append(img_)
    return inputs



def congate_(nparray,textfile,csvfile):
    text_ = open(textfile,'r')
    lines = text_.readlines()
    if(len(lines)!=nparray.shape[0]):
        print('ERROR! nparray do not have %d lines! Didn\'t congate.'%len(lines))
        return 0 
    arr = np.zeros((512,3),dtype = 'float64')
    index = 0
    for line in lines:
        image = line.split(' ')[0]
        image_num = int((image.split('.')[0]+'_').split('_')[0])
        arr[image_num] += nparray[index]
        index+=1
    arr/=(len(lines)/512.0)

    csv_file = open(csvfile,'w')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['image_name','Type_1','Type_2','Type_3'])
    for i in range(512):
        image_name = str(i)+'.jpg'
        data = [image_name,'%.16f'%arr[i][0], \
                '%.16f'%arr[i][1], \
                '%.16f'%arr[i][2]]
        csv_writer.writerow(data)
    text_.close()
    csv_file.close()


#  MEAN_PROTO_PATH = myself+'bvlc_googlenet/leveldb_and_mean/fr8_mean.binaryproto'
#  blob = caffe.proto.caffe_pb2.BlobProto()           
#  data = open(MEAN_PROTO_PATH, 'rb' ).read()        
#  blob.ParseFromString(data)                       
#  array = np.array(caffe.io.blobproto_to_array(blob))
#  mean_npy = array[0]  
#  mu=mean_npy.mean(1).mean(1)
#  print(mu)


def doClassify(model_def,model_weights,image_dims,mean_value,test_dir,test_txt):
    caffe.set_mode_gpu()
    np
    model_def = model_def
    model_weights = model_weights
    predictor = Classifier( model_def, model_weights, image_dims=(399,399), mean=np.array([97.5,91.5,121.2],dtype='float64'))
    inputs = make_inputs(test_dir, test_txt)
    pred_1,pred_2,pred_3 = predictor.predict(inputs,oversample=True)
    congate_(pred_1, test_txt, '/mnt/lustre/dingyang/cccc1.csv')
    congate_(pred_2, test_txt, '/mnt/lustre/dingyang/cccc2.csv')
    congate_(pred_3, test_txt, '/mnt/lustre/dingyang/cccc3.csv')
    print(pred_1)
    print(pred_2)
    print(pred_3)
    np.save('/mnt/lustre/dingyang/prob_2.npy',pred)
    caffe.mpi_fin()



model_def = myself+'inception-v4/inceptionV4_deploy_ADLoss_b1b7.prototxt' 
model_weights ='/mnt/lustre/dingyang/cervix_caffe_dingyang/myself/inception-v4/savedCaffemodels/iv4_ZXYBN_ADLoss_b1b7_jie_2_iter_300.caffemodel'
image_dims=(399,399)
mean_value=[97.5,91.5,121.2]
test_dir= '/mnt/lustre/dingyang/test_399x4/'
test_txt= '/mnt/lustre/dingyang/test_399x4.txt'
doClassify(model_def, model_weights, image_dims, mean_value, test_dir, test_txt)











