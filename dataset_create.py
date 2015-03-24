import cv
import cv2
import os
import theano
import glob
import cPickle
import pickle
import gzip
import numpy as np
from PIL import Image

Tr_c = glob.glob('your path .........../train_set/Cat*.pgm')
Tr_d = glob.glob('your path .........../train_set/Dog*.pgm')

train_set=[]

train_img1 = []
Trt_set = []

Trt_cat = 0
Trt_dog = 1

for i in range(number of training pics):
   
    img_cat = cv2.imread(Tr_c[i], cv2.CV_LOAD_IMAGE_GRAYSCALE)
    img_cat_r = cv2.resize(img_cat,(28, 28), interpolation = cv2.INTER_AREA)
    img_cat_arr = np.asarray(img_cat_r)
    img_cat_flo = np.float32(img_cat_arr)
    cv2.normalize(img_cat_flo,img_cat_flo,0,255,cv2.NORM_MINMAX)
    C = img_cat_flo/255
    
    img_dog = cv2.imread(Tr_d[i], cv2.CV_LOAD_IMAGE_GRAYSCALE)
    img_dog_r = cv2.resize(img_dog,(28, 28), interpolation = cv2.INTER_AREA)
    img_dog_arr = np.asarray(img_dog_r)
    img_dog_flo = np.float32(img_dog_arr)
    cv2.normalize(img_dog_flo,img_cat_flo,0,255,cv2.NORM_MINMAX)
    D = img_dog_flo/255
    
    C = C.reshape(1,-1)
    D = D.reshape(1,-1)
    
    train_img1.append(C)
    Trt_set.append(Trt_cat)
    
    train_img1.append(D)
    Trt_set.append(Trt_dog)

train_img1 = np.asarray(train_img1)
train_data_array = train_img1.reshape(120,784)
train_data_array

train_T_set = np.asarray(Trt_set)

train_set.append(train_data_array)
train_set.append(train_T_set)
train_set = tuple(train_set)

''Same code can be used to creat the tuple for validation and testing set.''
''The following code shows the method to create the pickled zip file like MNIST data set''

dataset = []
dataset.append(train_set)
dataset.append(valid_set)
dataset.append(test_set)

dataset = tuple(dataset)

f = file('file_name.pkl', 'wb')
cPickle.dump(dataset, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()
