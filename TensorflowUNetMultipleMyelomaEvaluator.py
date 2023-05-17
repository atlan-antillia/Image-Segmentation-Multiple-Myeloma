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

# TensorflowUNetMultipleMyelomaEvaluator.py
# 2023/05/10 to-arai


import os
import sys

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_ENABLE_GPU_GARBAGE_COLLECTION"]="false"

import traceback

from ConfigParser import ConfigParser
from MultipleMyelomaDataset import MultipleMyelomaDataset

from TensorflowUNet import TensorflowUNet
from GrayScaleImageWriter import GrayScaleImageWriter

MODEL  = "model"
TRAIN  = "train"
EVAL   = "eval"


if __name__ == "__main__":
  try:
    config_file    = "./train_eval_infer.config"
    # You can specify config_file on your command line parammeter.
    if len(sys.argv) == 2:
      cfile = sys.argv[1]
      if not os.path.exists(cfile):
         raise Exception("Not found " + cfile)
      else:
        config_file = cfile

    config     = ConfigParser(config_file)

    width      = config.get(MODEL, "image_width")
    height     = config.get(MODEL, "image_height")
    channels   = config.get(MODEL, "image_channels")
    if not (width == height and  height % 128 == 0 and width % 128 == 0):
      raise Exception("Image width should be a multiple of 128. For example 128, 256, 512")
    
    # Create a UNetMolde and compile
    model          = TensorflowUNet(config_file)
    
    # 1 Create test dataset
    resized_image    = (height, width, channels)
    dataset          = MultipleMyelomaDataset(resized_image)

    original_data_path  = config.get(EVAL, "image_datapath")
    segmented_data_path = config.get(EVAL, "mask_datapath")
   
    x_test, y_test = dataset.create(original_data_path, segmented_data_path)
    print(" len x_test {}".format(len(x_test)))
    print(" len y_test {}".format(len(y_test)))

    # 2 Create a UNetMolde and compile
    model          = TensorflowUNet(config_file)

    # 3 Start training
    model.evaluate(x_test, y_test)

  except:
    traceback.print_exc()
    
