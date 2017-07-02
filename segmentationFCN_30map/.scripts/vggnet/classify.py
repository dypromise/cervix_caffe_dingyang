import numpy as np
# The caffe module needs to be on the Python path;
#  we'll add it here explicitly.
import sys
caffe_root = '/mnt/lustre/cervix_caffe_dingyang/caffe/'  
myself='/mnt/lustre/cervix_caffe_dingyang/myself/'
sys.path.insert(0, caffe_root + 'python')

import caffe
import os
if os.path.isfile(myself+'bvlc_googlenet/bvlc_googlenet_iter_60000.caffemodel'):
    print 'GoogleNet found.'
else:
    print 'Don\'t found GoogleNet model.'

MEAN_PROTO_PATH = myself+'leveldb_and_mean/cervix_mean_binaryproto'
blob = caffe.proto.caffe_pb2.BlobProto()           
data = open(MEAN_PROTO_PATH, 'rb' ).read()        
blob.ParseFromString(data)                       

array = np.array(caffe.io.blobproto_to_array(blob))
mean_npy = array[0]                               

caffe.set_mode_cpu()
model_def = myself+'bvlc_googlenet/deploy.prototxt' 
model_weights =myself+'bvlc_googlenet/bvlc_googlenet_iter_80000.caffemodel'
net = caffe.Net(model_def,model_weights,caffe.TEST)     # use test mode (e.g., don't perform dropout)

# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = mean_npy.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print 'mean-subtracted values:', zip('BGR', mu)

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR


# set the size of the input (we can skip this if we're happy
#  with the default; we can also change it later, e.g., for different batch sizes)
#net.blobs['data'].reshape(50,3,256,256) 

#  image = caffe.io.load_image(caffe_root + 'examples/images/cat.jpg')
#  transformed_image = transformer.preprocess('data', image)
#  plt.imshow(image) 


import cv2
image_test_dir=myself+'test_data'
imagefiles = os.listdir(image_test_dir)
labels=["Type_1","Type_2","Type_3"]
for image in imagefiles:
    #img = cv2.imread(image)
    #print(img.shape)
    #  img_resized = cv2.resize(img,(256,256))
    print(image)
    img_resized = caffe.io.load_image(os.path.join(image_test_dir,image))
    #print(img_resized.shape)
    img_= transformer.preprocess('data',img_resized)
    print(img_.shape)
    net.blobs['data'].data[...]=img_
    # perform classification
    net.forward()
    # obtain the output probabilities
    output_prob = net.blobs['inception_5b/output'].data[0]
    print(output_prob)
    np.save('/mnt/lustre/dingyang/'+image+'_.npy',output_prob)
    # sort top five predictions from softmax output
    #top_inds = output_prob.argsort()[:3]
    print 'probabilities and labels:'
    #  for i in np.arange(top_inds.size):
        #  print top_inds[i], labels[top_inds[i]]
