# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import math
import os
import random
import shutil
import sys

sys.path.append('../../')

from libs.configs import cfgs


def mkdir(path):
  if not os.path.exists(path):
    os.makedirs(path)


divide_rate = 0.8

image_path = os.path.join(cfgs.ROOT_PATH, '{}/JPEGImages'.format(cfgs.DATASET_NAME))
xml_path = os.path.join(cfgs.ROOT_PATH, '{}/Annotations'.format(cfgs.DATASET_NAME))

image_list = os.listdir(image_path)

image_name = [n.split('.')[0] for n in image_list]

random.shuffle(image_name)

train_image = image_name[:int(math.ceil(len(image_name)) * divide_rate)]
test_image = image_name[int(math.ceil(len(image_name)) * divide_rate):]

image_output_train = os.path.join(
    cfgs.ROOT_PATH, '{}_train/JPEGImages'.format(cfgs.DATASET_NAME))
mkdir(image_output_train)
image_output_test = os.path.join(
    cfgs.ROOT_PATH, '{}_test/JPEGImages'.format(cfgs.DATASET_NAME))
mkdir(image_output_test)

xml_train = os.path.join(cfgs.ROOT_PATH, '{}_train/Annotations'.format(cfgs.DATASET_NAME))
mkdir(xml_train)
xml_test = os.path.join(cfgs.ROOT_PATH, '{}_test/Annotations'.format(cfgs.DATASET_NAME))
mkdir(xml_test)


count = 0
for i in train_image:
  shutil.copy(os.path.join(image_path, i + '.jpg'), image_output_train)
  if os.path.exists(os.path.join(xml_path, i + '.xml')):
    shutil.copy(os.path.join(xml_path, i + '.xml'), xml_train)
  if count % 1000 == 0:
    print("process step {}".format(count))
  count += 1

for i in test_image:
  shutil.copy(os.path.join(image_path, i + '.jpg'), image_output_test)
  shutil.copy(os.path.join(xml_path, i + '.xml'), xml_test)
  if count % 1000 == 0:
    print("process step {}".format(count))
  count += 1
