# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf

"""
cls : car|| Recall: 0.9542048293089093 || Precison: 0.002486730925298256|| AP: 0.8861557852184543
____________________
cls : aeroplane|| Recall: 0.9333333333333333 || Precison: 0.0005925041542114572|| AP: 0.842025283046165
____________________
cls : diningtable|| Recall: 0.8932038834951457 || Precison: 0.00040969553387335954|| AP: 0.7380756727563009
____________________
cls : cow|| Recall: 0.9672131147540983 || Precison: 0.0005127837421479989|| AP: 0.8598673192007477
____________________
cls : boat|| Recall: 0.870722433460076 || Precison: 0.0005048556531707801|| AP: 0.614924861322453
____________________
cls : person|| Recall: 0.9112190812720848 || Precison: 0.00890183387270766|| AP: 0.8403573910074297
____________________
cls : bottle|| Recall: 0.8550106609808102 || Precison: 0.0008528612324589202|| AP: 0.6431100249936452
____________________
cls : bus|| Recall: 0.9624413145539906 || Precison: 0.00045113940207524124|| AP: 0.8462278674877455
____________________
cls : chair|| Recall: 0.8465608465608465 || Precison: 0.0013795928045612787|| AP: 0.5683751395230314
____________________
cls : train|| Recall: 0.9361702127659575 || Precison: 0.0005837401825514753|| AP: 0.8364460305413468
____________________
cls : dog|| Recall: 0.9754601226993865 || Precison: 0.0010383537847668929|| AP: 0.9033364857501106
____________________
cls : motorbike|| Recall: 0.92 || Precison: 0.0006513139551094382|| AP: 0.8295629935581179
____________________
cls : cat|| Recall: 0.9608938547486033 || Precison: 0.0007397197236372707|| AP: 0.8871965890034861
____________________
cls : sofa|| Recall: 0.9539748953974896 || Precison: 0.0005096076691483964|| AP: 0.8040267075520897
____________________
cls : bird|| Recall: 0.9150326797385621 || Precison: 0.0009065830883400464|| AP: 0.8237404489482196
____________________
cls : horse|| Recall: 0.9568965517241379 || Precison: 0.0007317362585204424|| AP: 0.8735567893480496
____________________
cls : pottedplant|| Recall: 0.7895833333333333 || Precison: 0.0008196455411498826|| AP: 0.4870711669635938
____________________
cls : bicycle|| Recall: 0.9109792284866469 || Precison: 0.0006712950308861314|| AP: 0.8361496835573192
____________________
cls : tvmonitor|| Recall: 0.9512987012987013 || Precison: 0.0006329305331737686|| AP: 0.7944641602539069
____________________
cls : sheep|| Recall: 0.9214876033057852 || Precison: 0.0004898504308706817|| AP: 0.7823429444259854
____________________
mAP is : 0.7848506672229101    USE_12_METRIC

cls : tvmonitor|| Recall: 0.9512987012987013 || Precison: 0.0006329305331737686|| AP: 0.7680923930233127
____________________
cls : sheep|| Recall: 0.9214876033057852 || Precison: 0.0004898504308706817|| AP: 0.7605331894258903
____________________
cls : cat|| Recall: 0.9608938547486033 || Precison: 0.0007397197236372707|| AP: 0.8561115936556246
____________________
cls : train|| Recall: 0.9361702127659575 || Precison: 0.0005837401825514753|| AP: 0.7978443126776701
____________________
cls : aeroplane|| Recall: 0.9333333333333333 || Precison: 0.0005925041542114572|| AP: 0.8033498986573708
____________________
cls : motorbike|| Recall: 0.92 || Precison: 0.0006513139551094382|| AP: 0.7991141448766482
____________________
cls : bus|| Recall: 0.9624413145539906 || Precison: 0.00045113940207524124|| AP: 0.8181027979596772
____________________
cls : bird|| Recall: 0.9150326797385621 || Precison: 0.0009065830883400464|| AP: 0.7851247014320806
____________________
cls : pottedplant|| Recall: 0.7895833333333333 || Precison: 0.0008196455411498826|| AP: 0.49033575375142174
____________________
cls : cow|| Recall: 0.9672131147540983 || Precison: 0.0005127837421479989|| AP: 0.8304367006298838
____________________
cls : person|| Recall: 0.9112190812720848 || Precison: 0.00890183387270766|| AP: 0.797530517185023
____________________
cls : bottle|| Recall: 0.8550106609808102 || Precison: 0.0008528612324589202|| AP: 0.6320745719617634
____________________
cls : sofa|| Recall: 0.9539748953974896 || Precison: 0.0005096076691483964|| AP: 0.7868518534567335
____________________
cls : boat|| Recall: 0.870722433460076 || Precison: 0.0005048556531707801|| AP: 0.6036612374959088
____________________
cls : car|| Recall: 0.9542048293089093 || Precison: 0.002486730925298256|| AP: 0.8623955910304107
____________________
cls : bicycle|| Recall: 0.9109792284866469 || Precison: 0.0006712950308861314|| AP: 0.8029062441256611
____________________
cls : dog|| Recall: 0.9754601226993865 || Precison: 0.0010383537847668929|| AP: 0.8661350949646617
____________________
cls : diningtable|| Recall: 0.8932038834951457 || Precison: 0.00040969553387335954|| AP: 0.7127169697539509
____________________
cls : horse|| Recall: 0.9568965517241379 || Precison: 0.0007317362585204424|| AP: 0.8422342978325045
____________________
cls : chair|| Recall: 0.8465608465608465 || Precison: 0.0013795928045612787|| AP: 0.563170854101135
____________________
mAP is : 0.7589361358998666   USE_07_METRIC
"""

# ------------------------------------------------
VERSION = 'FPN_Res101_20181201_v2'
NET_NAME = 'resnet_v1_101'
ADD_BOX_IN_TENSORBOARD = True

# ---------------------------------------- System_config
ROOT_PATH = os.path.abspath('../')
print (20*"++--")
print (ROOT_PATH)
GPU_GROUP = "3"
SHOW_TRAIN_INFO_INTE = 10
SMRY_ITER = 100
SAVE_WEIGHTS_INTE = 10000

SUMMARY_PATH = ROOT_PATH + '/output/summary'
TEST_SAVE_PATH = ROOT_PATH + '/tools/test_result'
INFERENCE_IMAGE_PATH = ROOT_PATH + '/tools/inference_image'
INFERENCE_SAVE_PATH = ROOT_PATH + '/tools/inference_results'

if NET_NAME.startswith("resnet"):
    weights_name = NET_NAME
elif NET_NAME.startswith("MobilenetV2"):
    weights_name = "mobilenet/mobilenet_v2_1.0_224"
else:
    raise NotImplementedError

PRETRAINED_CKPT = ROOT_PATH + '/data/pretrained_weights/' + weights_name + '.ckpt'
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

EVALUATE_DIR = ROOT_PATH + '/output/evaluate_result_pickle/'
test_annotate_path = '/home/yjr/DataSet/VOC/VOC_test/VOC2007/Annotations'

# ------------------------------------------ Train config
RESTORE_FROM_RPN = False
IS_FILTER_OUTSIDE_BOXES = False
FIXED_BLOCKS = 0  # allow 0~3
USE_07_METRIC = False

RPN_LOCATION_LOSS_WEIGHT = 1.
RPN_CLASSIFICATION_LOSS_WEIGHT = 1.0

FAST_RCNN_LOCATION_LOSS_WEIGHT = 1.0
FAST_RCNN_CLASSIFICATION_LOSS_WEIGHT = 1.0
RPN_SIGMA = 3.0
FASTRCNN_SIGMA = 1.0

MUTILPY_BIAS_GRADIENT = None   # 2.0  # if None, will not multipy
GRADIENT_CLIPPING_BY_NORM = None   # 10.0  if None, will not clip

EPSILON = 1e-5
MOMENTUM = 0.9
LR = 0.001  # 0.001  # 0.0003
DECAY_STEP = [60000, 80000]  # 50000, 70000
MAX_ITERATION = 150000

# -------------------------------------------- Data_preprocess_config
DATASET_NAME = 'pascal'  # 'ship', 'spacenet', 'pascal', 'coco'
PIXEL_MEAN = [123.68, 116.779, 103.939]  # R, G, B. In tf, channel is RGB. In openCV, channel is BGR
IMG_SHORT_SIDE_LEN = 600  # 600  # 600
IMG_MAX_LENGTH = 1000  # 1000  # 1000
CLASS_NUM = 20

# --------------------------------------------- Network_config
BATCH_SIZE = 1
INITIALIZER = tf.random_normal_initializer(mean=0.0, stddev=0.01)
BBOX_INITIALIZER = tf.random_normal_initializer(mean=0.0, stddev=0.001)
WEIGHT_DECAY = 0.00004 if NET_NAME.startswith('Mobilenet') else 0.0001

# ---------------------------------------------Anchor config
USE_CENTER_OFFSET = False

LEVLES = ['P2', 'P3', 'P4', 'P5', 'P6']
BASE_ANCHOR_SIZE_LIST = [32, 64, 128, 256, 512]  # addjust the base anchor size for voc.
ANCHOR_STRIDE_LIST = [4, 8, 16, 32, 64]
ANCHOR_SCALES = [1.0]
ANCHOR_RATIOS = [0.5, 1., 2.0]
ROI_SCALE_FACTORS = [10., 10., 5.0, 5.0]
ANCHOR_SCALE_FACTORS = None

# --------------------------------------------FPN config
SHARE_HEADS = False
KERNEL_SIZE = 3
RPN_IOU_POSITIVE_THRESHOLD = 0.7
RPN_IOU_NEGATIVE_THRESHOLD = 0.3
TRAIN_RPN_CLOOBER_POSITIVES = False

RPN_MINIBATCH_SIZE = 256
RPN_POSITIVE_RATE = 0.5
RPN_NMS_IOU_THRESHOLD = 0.7
RPN_TOP_K_NMS_TRAIN = 12000
RPN_MAXIMUM_PROPOSAL_TARIN = 2000

RPN_TOP_K_NMS_TEST = 6000
RPN_MAXIMUM_PROPOSAL_TEST = 1000

# specific settings for FPN
# FPN_TOP_K_PER_LEVEL_TRAIN = 2000
# FPN_TOP_K_PER_LEVEL_TEST = 1000

# -------------------------------------------Fast-RCNN config
ROI_SIZE = 14
ROI_POOL_KERNEL_SIZE = 2
USE_DROPOUT = False
KEEP_PROB = 1.0
SHOW_SCORE_THRSHOLD = 0.5  # only show in tensorboard

FAST_RCNN_NMS_IOU_THRESHOLD = 0.3  # 0.6
FAST_RCNN_NMS_MAX_BOXES_PER_CLASS = 100
FAST_RCNN_IOU_POSITIVE_THRESHOLD = 0.5
FAST_RCNN_IOU_NEGATIVE_THRESHOLD = 0.0   # 0.1 < IOU < 0.5 is negative
FAST_RCNN_MINIBATCH_SIZE = 256  # if is -1, that is train with OHEM
FAST_RCNN_POSITIVE_RATE = 0.25

ADD_GTBOXES_TO_TRAIN = False



