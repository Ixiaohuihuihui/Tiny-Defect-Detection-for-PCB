# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf
'''
cls : aeroplane|| Recall: 0.9473684210526315 || Precison: 0.0006199030196164867|| AP: 0.826992691184208
____________________
cls : cow|| Recall: 0.9631147540983607 || Precison: 0.0005354526625668462|| AP: 0.8344652186720717
____________________
cls : dog|| Recall: 0.9652351738241309 || Precison: 0.0010528593384385115|| AP: 0.8848104631457077
____________________
cls : pottedplant|| Recall: 0.7708333333333334 || Precison: 0.000823124000293655|| AP: 0.4527288299945802
____________________
cls : diningtable|| Recall: 0.8980582524271845 || Precison: 0.00042887810125232407|| AP: 0.6700510019755388
____________________
cls : bird|| Recall: 0.9237472766884531 || Precison: 0.0009519832235409285|| AP: 0.7858394634006082
____________________
cls : tvmonitor|| Recall: 0.9415584415584416 || Precison: 0.0006423247726945524|| AP: 0.7532342429791412
____________________
cls : chair|| Recall: 0.8452380952380952 || Precison: 0.0014212159291838572|| AP: 0.5629849133883229
____________________
cls : train|| Recall: 0.925531914893617 || Precison: 0.0006022581218315107|| AP: 0.81368729431196
____________________
cls : horse|| Recall: 0.9454022988505747 || Precison: 0.0007562505602920185|| AP: 0.8603450848286776
____________________
cls : cat|| Recall: 0.9608938547486033 || Precison: 0.0007696817008175631|| AP: 0.8780370107529119
____________________
cls : sofa|| Recall: 0.9623430962343096 || Precison: 0.0005431753558979397|| AP: 0.748024582610825
____________________
cls : bottle|| Recall: 0.8571428571428571 || Precison: 0.0008744472166693132|| AP: 0.6253912291817303
____________________
cls : person|| Recall: 0.9149734982332155 || Precison: 0.009253199981238986|| AP: 0.8351147684067881
____________________
cls : car|| Recall: 0.9533721898417985 || Precison: 0.00259228066362385|| AP: 0.8841614814276471
____________________
cls : boat|| Recall: 0.8821292775665399 || Precison: 0.0005342347777629379|| AP: 0.6106671555293245
____________________
cls : motorbike|| Recall: 0.9323076923076923 || Precison: 0.0006731029825348658|| AP: 0.8421918380864666
____________________
cls : bicycle|| Recall: 0.9317507418397626 || Precison: 0.0007036524942688176|| AP: 0.8552669093308443
____________________
cls : bus|| Recall: 0.9765258215962441 || Precison: 0.00047651993823568495|| AP: 0.8420876315549962
____________________
cls : sheep|| Recall: 0.9049586776859504 || Precison: 0.000502333902950925|| AP: 0.7647489734437813
____________________
mAP is : 0.7665415392103065   USE_12_METRIC

cls : bicycle|| Recall: 0.9317507418397626 || Precison: 0.0007036524942688176|| AP: 0.8298982119397122
____________________
cls : sofa|| Recall: 0.9623430962343096 || Precison: 0.0005431753558979397|| AP: 0.7272523895735249
____________________
cls : bus|| Recall: 0.9765258215962441 || Precison: 0.00047651993823568495|| AP: 0.8137027123104137
____________________
cls : diningtable|| Recall: 0.8980582524271845 || Precison: 0.00042887810125232407|| AP: 0.6530525394835751
____________________
cls : person|| Recall: 0.9149734982332155 || Precison: 0.009253199981238986|| AP: 0.803256081733613
____________________
cls : car|| Recall: 0.9533721898417985 || Precison: 0.00259228066362385|| AP: 0.8577825832291308
____________________
cls : boat|| Recall: 0.8821292775665399 || Precison: 0.0005342347777629379|| AP: 0.5979719282542533
____________________
cls : chair|| Recall: 0.8452380952380952 || Precison: 0.0014212159291838572|| AP: 0.5599343653732526
____________________
cls : aeroplane|| Recall: 0.9473684210526315 || Precison: 0.0006199030196164867|| AP: 0.7917730109896329
____________________
cls : cat|| Recall: 0.9608938547486033 || Precison: 0.0007696817008175631|| AP: 0.8475644227001603
____________________
cls : sheep|| Recall: 0.9049586776859504 || Precison: 0.000502333902950925|| AP: 0.7327379110779253
____________________
cls : train|| Recall: 0.925531914893617 || Precison: 0.0006022581218315107|| AP: 0.7743045860493956
____________________
cls : horse|| Recall: 0.9454022988505747 || Precison: 0.0007562505602920185|| AP: 0.8223412836194737
____________________
cls : cow|| Recall: 0.9631147540983607 || Precison: 0.0005354526625668462|| AP: 0.8058877343148467
____________________
cls : tvmonitor|| Recall: 0.9415584415584416 || Precison: 0.0006423247726945524|| AP: 0.7310441973657807
____________________
cls : pottedplant|| Recall: 0.7708333333333334 || Precison: 0.000823124000293655|| AP: 0.4646864671975241
____________________
cls : dog|| Recall: 0.9652351738241309 || Precison: 0.0010528593384385115|| AP: 0.8525619478862897
____________________
cls : bird|| Recall: 0.9237472766884531 || Precison: 0.0009519832235409285|| AP: 0.7610720209528306
____________________
cls : bottle|| Recall: 0.8571428571428571 || Precison: 0.0008744472166693132|| AP: 0.6127328834288011
____________________
cls : motorbike|| Recall: 0.9323076923076923 || Precison: 0.0006731029825348658|| AP: 0.8119378019468331
____________________
mAP is : 0.7425747539713485    USE_07_METRIC

'''

# ------------------------------------------------
VERSION = 'FPN_Res50_20181201'
NET_NAME = 'resnet_v1_50'
ADD_BOX_IN_TENSORBOARD = True

# ---------------------------------------- System_config
ROOT_PATH = os.path.abspath('../')
print (20*"++--")
print (ROOT_PATH)
GPU_GROUP = "1"
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
SHARE_HEADS = True
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



