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
# We have taken the original segmentation dataset for Brain MRI from
# LGG Segmentation Dataset
# https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation

"""
About Dataset
LGG Segmentation Dataset
Dataset used in:

Mateusz Buda, AshirbaniSaha, Maciej A. Mazurowski "Association of genomic subtypes of lower-grade gliomas 
with shape features automatically extracted by a deep learning algorithm." Computers in Biology and Medicine, 2019.

and

Maciej A. Mazurowski, Kal Clark, Nicholas M. Czarnek, Parisa Shamsesfandabadi, Katherine B. Peters, Ashirbani Saha 
"Radiogenomics of lower-grade glioma: algorithmically-assessed tumor shape is associated with tumor genomic subtypes
 and patient outcomes in a multi-institutional study with The Cancer Genome Atlas data." Journal of Neuro-Oncology, 2017.

This dataset contains brain MR images together with manual FLAIR abnormality segmentation masks.
The images were obtained from The Cancer Imaging Archive (TCIA).
They correspond to 110 patients included in The Cancer Genome Atlas (TCGA) lower-grade glioma collection with at
 least fluid-attenuated inversion recovery (FLAIR) sequence and genomic cluster data available.
Tumor genomic clusters and patient data is provided in data.csv file.
"""

# split_master.py

import os
import glob
import shutil
import traceback
import numpy as np

import cv2
import random

def split_master(input_dir, train_dir, test_dir, save_as_jpg=False):
  valid_mask_files = listup_valid_mask_files(input_dir)
  random.shuffle(valid_mask_files)
  num_files  = len(valid_mask_files)
  num_train  = int (num_files * 0.8)
  num_test   = int (num_files * 0.2)

  mask_train_files = valid_mask_files[0: num_train]
  mask_test_files  = valid_mask_files[num_train: num_files]
  print(" num files {}".format(num_files))
  print(" num train {}".format(num_train))
  print(" num test  {}".format(num_test))
  dataset_dirs = [train_dir, test_dir]
  mask_files   = [mask_train_files, mask_test_files]
  for i, dataset_dir in enumerate(dataset_dirs):
     
    output_mask_dir  = os.path.join(dataset_dir, "mask")
    output_image_dir = os.path.join(dataset_dir, "image")
    if not os.path.exists(output_mask_dir):
      os.makedirs(output_mask_dir)
    if not os.path.exists(output_image_dir):
      os.makedirs(output_image_dir)

    for mask_file in mask_files[i]:
      image_file = mask_file.replace("_mask", "")

      if save_as_jpg:     
        save_as_jpg(mask_file, output_mask_dir)
        save_as_jpg(image_file, output_image_dir)
      else:
        shutil.copy2(mask_file, output_mask_dir)
        print("---Copied {} to {}".format(mask_file, output_mask_dir))
        shutil.copy2(image_file, output_image_dir)
        print("---Copied {} to {}".format(image_file, output_image_dir))

def save_as_jpg(image_file, output_dir):
  img = cv2.imread(image_file, cv2.COLOR_BGR2RGB)
  basename = os.path.basename(image_file)
  name     = basename.split(".")[0]
  output_filepath = os.path.join(output_dir, name + ".jpg")
  cv2.imwrite(output_filepath, img, [cv2.IMWRITE_JPEG_QUALITY, 95]))


def listup_valid_mask_files(input_dir):
  dirs = os.listdir(input_dir)
  valid_mask_files = []
  for dir in dirs:
    input_subdir = os.path.join(input_dir, dir)
    if os.path.isdir(input_subdir):
      mask_pattern = input_subdir + "/*_mask.tif"
      mask_files   = glob.glob(mask_pattern)
      for mask_file in mask_files:
        image_file = mask_file.replace("_mask", "")
        mask = cv2.imread(mask_file)
        
        if np.any(mask==(255, 255, 255)):
          valid_mask_files.append(mask_file)
        else:
          pass
  return valid_mask_files     
    

if __name__ == "__main__":
  try:
    input_dir  = "./lgg-mri-segmentation/kaggle_3m"
    output_dir = "./BrainTumor/"
    train_dir = output_dir + "train"
    test_dir  = output_dir + "test"

    if os.path.exists(train_dir):
      shutil.rmtree(train_dir)
    if not os.path.exists(train_dir):
      os.makedirs(train_dir)
 
    if os.path.exists(test_dir):
      shutil.rmtree(test_dir)
    if not os.path.exists(test_dir):
      os.makedirs(test_dir)
    
    split_master(input_dir, train_dir, test_dir, save_as_jpg=False)

  except:
    traceback.print_exc()

