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
import numpy as np
import cv2
from tqdm import tqdm
import glob
from matplotlib import pyplot as plt
from skimage.io import imread, imshow
import traceback

class MultipleMyelomaDataset:

  def __init__(self, resized_image, threshold=160):
    self.IMG_WIDTH, self.IMG_HEIGHT, self.IMG_CHANNELS = resized_image
    self.THRESHOLD = threshold
    self.BLUR_SIZE = (3, 3)

  def create(self, image_data_path, mask_data_path, has_mask=True, debug=False):
 
    image_files = sorted(glob.glob(image_data_path + "/*.jpg"))
    mask_files  = sorted(glob.glob(mask_data_path  + "/*.jpg"))
  
    num_images  = len(image_files)
    X = np.zeros((num_images, self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS), dtype=np.uint8)
    Y = np.zeros((num_images, self.IMG_HEIGHT, self.IMG_WIDTH, 1                ), dtype=np.bool)
    blur_size= (3, 3)
    for n, image_file in tqdm(enumerate(image_files), total=len(image_files)):
  
      image = cv2.imread(image_file, cv2.COLOR_BGR2RGB)
      image = cv2.resize(image, dsize= (self.IMG_HEIGHT, self.IMG_WIDTH), interpolation=cv2.INTER_NEAREST)
      X[n]  = image

      mask  = cv2.imread(mask_files[n])
      mask  = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
      mask  = cv2.resize(mask, dsize= (self.IMG_HEIGHT, self.IMG_WIDTH),   interpolation=cv2.INTER_NEAREST)

      # Binarize mask
      mask[mask< self.THRESHOLD] =   0  
      mask[mask>=self.THRESHOLD] = 255
      # Blur mask 
      mask = cv2.blur(mask, self.BLUR_SIZE)
  
      mask  = np.expand_dims(mask, axis=-1)
      Y[n]  = mask

      if debug:
          imshow(mask)
          plt.show()
          input("XX")   
  
    return X, Y


if __name__ == "__main__":
  try:
    resized_image = (128, 128, 3)
    
    train_image_datapath = "./MultipleMyeloma/train/images/"
    train_mask_datapath  = "./MultipleMyeloma/train/masks/"
    test_image_datapath  = "./MultipleMyeloma/valid/images/"
    test_mask_datapath   = "./MultipleMyeloma/valid/masks/"

    dataset = MultipleMyelomaDataset(resized_image)
    x_train, y_train = dataset.create(train_image_datapath, train_mask_datapath, has_mask=True, debug=True)
    print(" len x_train {}".format(len(x_train)))
    print(" len y_train {}".format(len(y_train)))

    x_test, y_test   = dataset.create(test_image_datapath, test_mask_datapath, has_mask=True)
    print(" len x_test {}".format(len(x_test)))
    print(" len y_test {}".format(len(y_test)))

  except:
    traceback.print_exc()

