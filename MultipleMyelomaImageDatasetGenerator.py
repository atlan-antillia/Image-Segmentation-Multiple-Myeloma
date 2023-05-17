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
# MultipleMyelomaImageDataGenerator.py
# 2023/05/18 : Toshiyuki Arai antillia.com

#    
#from email.mime import image
from ctypes import c_byte
import sys
import os
import glob
import random
import shutil
import numpy as np

import traceback
import cv2
from PIL import Image, ImageDraw, ImageFilter


class MultipleMyelomaImageDatasetGenerator:
  def __init__(self, W=256, H=256):
    self.W = W
    self.H = H
    self.backgrounds = []

  # dir = "./train/x
  # target = "./train" "./valid"
  def get_image_filepaths(self, images_dir ="./train/x"):
    pattern = images_dir + "/*.bmp"
    print("--- pattern {}".format(pattern))
    all_files  = glob.glob(pattern)
    image_filepaths = []
    for file in all_files:
      basename = os.path.basename(file)
      if basename.find("_") == -1:
        image_filepaths.append(file)
    return image_filepaths

  def get_mask_filepaths(self, image_filepath, mask_dir):
    basename = os.path.basename(image_filepath)
    name     = basename.split(".")[0]
    mask_filepattern  = mask_dir + "/" + name + "_*.bmp"
    mask_filepaths    = glob.glob(mask_filepattern)
    return mask_filepaths

  def create_backgrounds(self, image_filepaths, num):
    background_files = random.sample(image_filepaths, num)
    for background_file in background_files:
      img = Image.open(background_file)
      img = img.resize((self.W, self.H))
      
      blurred = img.filter(filter=ImageFilter.BLUR)
      self.backgrounds.append(blurred)

  # target = "./train" or "./validation"
  def create(self, input_dir, output_dir, crop_ellipse=False, debug=False):
    images_dir = input_dir + "/x/"
    masks_dir  = input_dir + "/y/"
    image_filepaths  = self.get_image_filepaths(images_dir)
    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)

    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    
    self.create_backgrounds(image_filepaths, 20)

    output_images_dir = os.path.join(output_dir, "images")
    output_masks_dir = os.path.join(output_dir, "masks")
    if not os.path.exists(output_images_dir):
      os.makedirs(output_images_dir)
    if not os.path.exists(output_masks_dir):
      os.makedirs(output_masks_dir)
    
    for image_filepath in image_filepaths:
      basename = os.path.basename(image_filepath)
      name     = basename.split(".")[0]

      # 1 Create resize_image of size 256x256
      img_256x256 = self.create_resized_images(image_filepath, mask=False)
      
      output_img_filepath = os.path.join(output_dir, name + ".jpg")
      # 2 Save the img_256x256 as a jpg file.
      #img_256x256.save(output_img_filepath)
      print("=== Saved image_filepath {} as {}".format(image_filepath, output_img_filepath))

      # 3 Get some mask_filepaths corresponding to the image_filepath
      mask_filepaths = self.get_mask_filepaths(image_filepath, masks_dir)

      pixel = None
      for mask_filepath in mask_filepaths:
        mask_basename = os.path.basename(mask_filepath)
        print(mask_basename)
        mask_filename   = mask_basename.split(".")[0]
        print("-------mask_filename {}".format(mask_filename))
        # 4 Create mask_image of size 256x256
        print("=== Create mask_image_256x256 from {}".format(mask_filepath))
        #PIL image format
        mask_img_256x256   = self.create_resized_images(mask_filepath, mask=True)
        
        
        # 5 get bounding box
      
        (x, y, w, h)  = self.get_boundinbox(mask_img_256x256)
        print(" x {} y {} w {} h {}".format(x, y, w, h))
        #cropped = img_256x256.crop((x, y, x+w, y+h))
        # Expand cripping region
        margin = 6
        cx = x -margin
        cy = y -margin
        if cx < 0:
           cx = 0
        if cy < 0:
           cy = 0
        cxr = cx + w + margin*2
        cyr = cy + h + margin*2
        if cxr >=self.W:
           cxr = self.W -1
        if cyr >=self.H:
           cyr = self.H -1
        print(" cx {} cy {} cxr {} cyr {}".format(cx, cy, cxr, cyr))
        
       
        cropped = img_256x256.crop((cx, cy, cxr, cyr))
        mask_crop = None
        if crop_ellipse:
          mask_crop = self.crop_ellipse(cropped)
        #blurred_mask = mask_c.filter(ImageFilter.GaussianBlur(10))
        #cropped = cropped.convert("RGB")
        #cropped.show()

        #cropped = self.crop_ellipse(img_256x256, (cx, cy), (cxr, cyr))
        print("----  cropped.size {}".format(cropped.size))
       
        #background = random.choice(self.backgrounds)
        #background = background.copy()
        pixel = (207, 196, 208)

        #if pixel == None:
        #  pixel = cropped.getpixel((2,2))
        background = Image.new("RGB", (self.W, self.H), pixel,) 

        #background = background.convert("RGBA")
        #background.show()
        #input("---------------------")
        
        background.paste(cropped, (cx, cy), mask_crop)
        
        output_image_filepath = os.path.join(output_images_dir, mask_filename + ".jpg")
      
        background.save(output_image_filepath)
        print("=== Saved cropped image {}".format(output_image_filepath))
      
        output_mask_filepath =  os.path.join(output_masks_dir, mask_filename + ".jpg")
        mask_img_256x256.save(output_mask_filepath)
        print("=== Save mask_image     {}".format(output_mask_filepath))


 
  def crop_ellipse(self, img):
     img = img.convert("RGB")  
     height,width = img.size
     mask = Image.new('L', [height,width] , 0)
  
     draw = ImageDraw.Draw(mask)
   
     draw.ellipse([(0,0), (height,width)], fill=255) #, outline="white")
     #mask = mask.filter(ImageFilter.GaussianBlur(10))

     img_arr  = np.array(img)
     mask_arr = np.array(mask)

     final_img_arr = np.dstack((img_arr, mask_arr))
     final_img_arr = final_img_arr.copy()
     cropped = Image.fromarray(final_img_arr)
     return cropped


  # Create a resized_256x256_image from each original file in image_filepaths
  def create_resized_images(self, image_filepath, mask=False):

    img = Image.open(image_filepath)
    print("---create_resized_256x256_images {}".format(image_filepath))
    
    #pixel = img.getpixel((128, 128))
    # We use the following fixed pixel for a background image.
    pixel = (207, 196, 208)
    pixel = (200, 180, 180)
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

   
    background_256x256 = background.resize((self.W, self.H))
    if mask:
      background_256x256 = self.convert2WhiteMask(background_256x256)

    return background_256x256

  # 2023.05/15
  def get_boundinbox(self, pil_mask_img_256x256):
        mask_img = np.array(pil_mask_img_256x256)

        mask_img= cv2.cvtColor(mask_img,  cv2.COLOR_RGB2GRAY)
      
        H, W = mask_img.shape[:2]
       
        contours, hierarchy = cv2.findContours(mask_img, 
           cv2.RETR_EXTERNAL, 
           cv2.CHAIN_APPROX_SIMPLE)
       
        contours = max(contours, key=lambda x: cv2.contourArea(x))
        x, y, w, h = cv2.boundingRect(contours)
        print("---x {} y {} w {} h {}".format(x, y, w, h))
        #Compute bouding box of YOLO format.
        return (x, y, w, h)
  
    
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
./MultipleMyeloma
├─train
└─valid
 
"""

"""
categories [0]
 
"MultipleMyeloma"        = 0



"""
if __name__ == "__main__":
  try:      
    # create Ovrian UltraSound Images OUS_augmented_master_256x256 dataset train, valid 
    # from the orignal Dataset_.

    input_dir   = "./TCIA_SegPC_dataset"
    # For simplicity, we have renamed the folder name from the original "validation" to "valid" 
    datasets    = ["train", "valid"]
    output_dir  = "./MultipleMyeloma"

    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    generator = MultipleMyelomaImageDatasetGenerator(W=256, H=256)
    debug = True
    crop_ellipse = False
    for dataset in datasets:
      input_subdir  = os.path.join(input_dir, dataset)
      output_subdir = os.path.join(output_dir, dataset)

      generator.create(input_subdir, output_subdir, crop_ellipse=crop_ellipse, debug=debug)
  except:
    traceback.print_exc()

      