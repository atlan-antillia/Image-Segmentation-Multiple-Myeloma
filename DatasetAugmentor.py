# Copyright 2023 (C) antillia.com. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#
# DatasetAugmentor.py
# 2023/04/23 : Toshiyuki Arai antillia.com

#    
#from email.mime import image
import sys
import os
import glob
import random
import shutil
import numpy as np

import traceback
import cv2
from PIL import Image


class DatasetAugmentor:
  def __init__(self, W=512, H=512):
    self.W = W
    self.H = H


  # target = "./train" or "./valid"
  def generate(self, input_dir, output_dir, augment=True):
    SUBDIRS = ["/x/", "/y/"]
    MASKS   = [False, True]

    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)

    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    
    for i, subdir in enumerate(SUBDIRS):
      print("---subdir {}".format(subdir))
      print("---input_dir {}".format(input_dir))
      images_subdir = input_dir  + subdir #os.path.join(input_dir, subdir) 
      output_subdir = output_dir + subdir #os.path.join(output_dir, subdir)
      print("--- images_subdir {}".format(images_subdir))
      print("--- output_subdir {}".format(output_subdir))
      #input("---HiT")
      if not os.path.exists(output_subdir):
        os.makedirs(output_subdir)
      
      pattern = images_subdir + "/*.bmp"
      image_filepaths = glob.glob(pattern)
      mask = MASKS[i]
      image_format    = "jpg"
      print("=== image_filepaths {}".format(image_filepaths))

      for image_filepath in image_filepaths:
        basename = os.path.basename(image_filepath)
        name     = basename.split(".")[0]

        # 1 Create resize_image of size 512x512
        img_512x512 = self.create_resized_images(image_filepath, mask=mask)     

        # 2 Create rotated images from the resized img_512x512
        self.create_rotated_image(img_512x512, name, image_format, output_subdir, augment=augment)


  # Create a resized_512x512_image from each original file in image_filepaths
  def create_resized_images(self, image_filepath, mask=False):

    img = Image.open(image_filepath)
    print("---create_resized_512x512_images {}".format(image_filepath))
    
    #pixel = img.getpixel((128, 128))
    # We use the following fixed pixel for a background image.
    pixel = (207, 196, 208)
    if mask:
      pixel = (0, 0, 0)
    print("----pixel {}".format(pixel))
    w, h = img.size
    max = w
    if h > w:
      max = h
    if max < self.W:
      max = self.W
    # 1 Create a black background image
    background = Image.new("RGB", (max, max), pixel) # (0, 0, 0))
    #input("----HIT")
    # 2 Paste the original img to the background image at (x, y) position.
    print(img.format, img.size, img.mode)
    print(background.format, background.size, background.mode)

    x = int( (max - w)/2 )
    y = int( (max - h)/2 )
    background.paste(img, (x, y))

   
    background_512x512 = background.resize((self.W, self.H))
    if mask:
      background_512x512 = self.convert2WhiteMask(background_512x512)

    return background_512x512



  def create_rotated_image(self, resized_img, nameonly, image_format, output_dir, augment=False):
    W, H = resized_img.size
    # Convert pil resized_img to resized_img of OpenCV
    resized_img = cv2.cvtColor(np.array(resized_img), cv2.COLOR_RGB2BGR)

    #ANGLES = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    #ANGLES = [0, 50, 100, 150, 200, 250, 300, 350]
    ANGLES = [0, 90, 180, 270]
    DELIMITER = "--"
    if augment == False:
      ANGLES = [0]

    for angle in ANGLES:
      center = (self.W/2, self.H/2)
      rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)

      # 3 Rotate the resize_mask_img by angle
      rotated_resized_image    = cv2.warpAffine(src=resized_img, M=rotate_matrix, dsize=(self.W, self.H))
      rotated_resized_filename = "rotated-" + str(angle) + DELIMITER + nameonly + "." + image_format
      rotated_resized_filepath = os.path.join(output_dir, rotated_resized_filename)

      # 5 Write the rotated_resized_mask_image as a jpg file.
      cv2.imwrite(rotated_resized_filepath, rotated_resized_image)
      print("Saved {} ".format(rotated_resized_filepath))

    if augment==False:
      return 
    """
    FLIPCODES = [0, 1]
    for flipcode in FLIPCODES:
      # 7 Flip the resized_mask_img by flipcode
      flipped_resized_img = cv2.flip(resized_img, flipcode)
      # Save flipped mask_filename is jpg
      save_flipped_img_filename = "flipped-" + str(flipcode) +  DELIMITER + nameonly + "." + image_format
      flipped_resized_img_filepath = os.path.join(output_dir, save_flipped_img_filename )

      # 8 Write the flipped_resized_mask_img as a jpg file.
      cv2.imwrite(flipped_resized_img_filepath, flipped_resized_img)
      print("Saved {} ".format(flipped_resized_img_filepath))
    """

    
  def convert2WhiteMask(self, image):
    w, h = image.size
    for y in range(h):
      for x in range(w):
        pixel = image.getpixel((x, y))
        if pixel != (0, 0, 0):
          pixel = (255, 255, 255) #White
          image.putpixel((x, y), pixel) 
    return image


"""
INPUT:

./TCIA_SegPC_dataset
├─train
└─valid


Output:
./YOLO
├─train
└─valid


categories [0]
"MultipleMyeloma"        = 0
"""

if __name__ == "__main__":
  try:      
    # create Ovrian UltraSound Images OUS_augmented_master_512x512 dataset train, valid 
    # from the orignal Dataset_.

    input_dir   = "./TCIA_SegPC_dataset/"
    # For simplicity, we have renamed the folder name from the original "validation" to "valid" 
    datasets    = ["train", "valid"]
  
    augments    = [True, False]
    output_dir  = "./MultipleMyeloma"

    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    annotation= DatasetAugmentor(W=512, H=512)
  
    for i, dataset in enumerate(datasets):
      input_subdir  = os.path.join(input_dir, dataset)
      output_subdir = os.path.join(output_dir, dataset)
      augment = augments[i]
      print("---input_subdir {}".format(input_subdir))
      annotation.generate(input_subdir, output_subdir, augment=augment)
  except:
    traceback.print_exc()

      