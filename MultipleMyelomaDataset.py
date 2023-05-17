# Copyright 2023 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# 2018 Data Science Bowl
# Find the nuclei in divergent images to advance medical discovery
# https://www.kaggle.com/c/data-science-bowl-2018

# This code is based on the Python scripts of the following web sites.
#
# 1. Keras U-Net starter - LB 0.277
# https://www.kaggle.com/code/keegil/keras-u-net-starter-lb-0-277/notebook
# 2. U-Net Image Segmentation in Keras
# https://androidkt.com/tensorflow-keras-unet-for-image-image-segmentation/


import os
import sys
import glob
import numpy as np
from matplotlib import pyplot as plt

#from tqdm import tqdm

# pip install scikit-image
from skimage.transform import resize
#from skimage.morphology import label
from skimage.io import imread, imshow

import traceback

class MultipleMyelomaDataset:

  def __init__(self, resized_image):
    self.IMG_WIDTH, self.IMG_HEIGHT, self.IMG_CHANNELS = resized_image

 
  def create(self, data_path, has_mask=True, debug=False):
    image_dir = "/images/"
    mask_dir  = "/masks/"
    image_files = sorted(glob.glob(data_path + image_dir + "*.jpg"))
    mask_files  = sorted(glob.glob(data_path + mask_dir + "*.jpg"))
    #mask_files  = sorted(glob.glob(data_path + mask_dir + "*.jpg"))
  
    num_images  = len(image_files)
    X = np.zeros((num_images, self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS), dtype=np.uint8)
    Y = np.zeros((num_images, self.IMG_HEIGHT, self.IMG_WIDTH, 1                ), dtype=np.bool)

    for n in range(num_images):
      image_file = image_files[n]
      image = imread(image_file)
      #print("--- image_file {}".format(image_file))
      image = resize(image, (self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS), mode='constant', preserve_range=True)
      X[n]  = image
  
      mask = imread(mask_files[n]) #, as_gray=True)
      #mask = resize(mask, (self.IMG_HEIGHT, self.IMG_WIDTH, 1), preserve_range=False, anti_aliasing=False) 
      mask = resize(mask, (self.IMG_HEIGHT, self.IMG_WIDTH, 1), mode='constant', preserve_range=False, anti_aliasing=False) 

                                      
      Y[n] = mask
      if debug:
          imshow(mask)
          plt.show()
          input("XX")   
  
    return X, Y


if __name__ == "__main__":
  try:
    resized_image = (128, 128, 3)
    
    train_datapath = "./MultipleMyeloma/train/"
    test_datapath  = "./MultipleMyeloma/valid/"

    dataset = MultipleMyelomaDataset(resized_image)
    x_train, y_train = dataset.create(train_datapath, has_mask=True, debug=False)
    print(" len x_train {}".format(len(x_train)))
    print(" len y_train {}".format(len(y_train)))

    x_test, y_test   = dataset.create(test_datapath, has_mask=True)
    print(" len x_test {}".format(len(x_test)))
    print(" len y_test {}".format(len(y_test)))


  except:
    traceback.print_exc()

