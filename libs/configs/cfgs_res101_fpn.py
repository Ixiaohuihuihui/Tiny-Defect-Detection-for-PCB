# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf
'''
cls : person|| Recall: 0.9200530035335689 || Precison: 0.009050166947990866|| AP: 0.8413662097687251
____________________
cls : cat|| Recall: 0.9608938547486033 || Precison: 0.0007414480221227399|| AP: 0.886410933036462
____________________
cls : horse|| Recall: 0.9626436781609196 || Precison: 0.0007370072226707821|| AP: 0.880462817781879
____________________
cls : boat|| Recall: 0.8783269961977186 || Precison: 0.000509740231082238|| AP: 0.6456185469835614
____________________
cls : bottle|| Recall: 0.8656716417910447 || Precison: 0.0008665714718695106|| AP: 0.6480626365413494
____________________
cls : bicycle|| Recall: 0.9228486646884273 || Precison: 0.0006832296772124229|| AP: 0.8550508887926864
____________________
cls : bus|| Recall: 0.9577464788732394 || Precison: 0.00045156820340048565|| AP: 0.8631526839193041
____________________
cls : sheep|| Recall: 0.9132231404958677 || Precison: 0.0004864327094081809|| AP: 0.7741568397678364
____________________
cls : car|| Recall: 0.9600333055786844 || Precison: 0.0025449055537652685|| AP: 0.8914023804170609
____________________
cls : motorbike|| Recall: 0.9538461538461539 || Precison: 0.0006737519288865706|| AP: 0.8495072139551133
____________________
cls : chair|| Recall: 0.873015873015873 || Precison: 0.001433311906044233|| AP: 0.5759698175528438
____________________
cls : aeroplane|| Recall: 0.9438596491228071 || Precison: 0.0006024690030817745|| AP: 0.8353670052573003
____________________
cls : tvmonitor|| Recall: 0.9415584415584416 || Precison: 0.0006278089036291684|| AP: 0.7613581746623427
____________________
cls : sofa|| Recall: 0.9707112970711297 || Precison: 0.0005222178954168627|| AP: 0.7987407803525022
____________________
cls : bird|| Recall: 0.9281045751633987 || Precison: 0.0009227090390830091|| AP: 0.8159836473038345
____________________
cls : dog|| Recall: 0.9611451942740287 || Precison: 0.0010255447933963644|| AP: 0.8951754362513265
____________________
cls : cow|| Recall: 0.9467213114754098 || Precison: 0.0005056795917786568|| AP: 0.8497306852549179
____________________
cls : diningtable|| Recall: 0.883495145631068 || Precison: 0.0004040099093419522|| AP: 0.7307392356452687
____________________
cls : pottedplant|| Recall: 0.7729166666666667 || Precison: 0.0008064077902035582|| AP: 0.4738700691112566
____________________
cls : train|| Recall: 0.9290780141843972 || Precison: 0.0005804331981204598|| AP: 0.8427500500899303
____________________
mAP is : 0.7857438026222752   (USE_12_METRIC)

cls : train|| Recall: 0.9290780141843972 || Precison: 0.0005804331981204598|| AP: 0.8101152343436091
____________________
cls : bus|| Recall: 0.9577464788732394 || Precison: 0.00045156820340048565|| AP: 0.830722622273239
____________________
cls : chair|| Recall: 0.873015873015873 || Precison: 0.001433311906044233|| AP: 0.5698849842652579
____________________
cls : pottedplant|| Recall: 0.7729166666666667 || Precison: 0.0008064077902035582|| AP: 0.48047763621440476
____________________
cls : horse|| Recall: 0.9626436781609196 || Precison: 0.0007370072226707821|| AP: 0.8512804991519783
____________________
cls : person|| Recall: 0.9200530035335689 || Precison: 0.009050166947990866|| AP: 0.8107708491164711
____________________
cls : bottle|| Recall: 0.8656716417910447 || Precison: 0.0008665714718695106|| AP: 0.63789938616088
____________________
cls : bicycle|| Recall: 0.9228486646884273 || Precison: 0.0006832296772124229|| AP: 0.8166723684624742
____________________
cls : dog|| Recall: 0.9611451942740287 || Precison: 0.0010255447933963644|| AP: 0.864470680916449
____________________
cls : diningtable|| Recall: 0.883495145631068 || Precison: 0.0004040099093419522|| AP: 0.7122255048941863
____________________
cls : bird|| Recall: 0.9281045751633987 || Precison: 0.0009227090390830091|| AP: 0.7832546811459113
____________________
cls : sofa|| Recall: 0.9707112970711297 || Precison: 0.0005222178954168627|| AP: 0.778305908921783
____________________
cls : sheep|| Recall: 0.9132231404958677 || Precison: 0.0004864327094081809|| AP: 0.7463330859344937
____________________
cls : boat|| Recall: 0.8783269961977186 || Precison: 0.000509740231082238|| AP: 0.6291419623367831
____________________
cls : car|| Recall: 0.9600333055786844 || Precison: 0.0025449055537652685|| AP: 0.8630428431995184
____________________
cls : motorbike|| Recall: 0.9538461538461539 || Precison: 0.0006737519288865706|| AP: 0.8224280778332824
____________________
cls : aeroplane|| Recall: 0.9438596491228071 || Precison: 0.0006024690030817745|| AP: 0.8001448356711514
____________________
cls : cat|| Recall: 0.9608938547486033 || Precison: 0.0007414480221227399|| AP: 0.8582414148566436
____________________
cls : cow|| Recall: 0.9467213114754098 || Precison: 0.0005056795917786568|| AP: 0.8242904910827928
____________________
cls : tvmonitor|| Recall: 0.9415584415584416 || Precison: 0.0006278089036291684|| AP: 0.7388745216642896
____________________
mAP is : 0.76142887942228   (USE_07_METRIC)

'''

# ------------------------------------------------
VERSION = 'FPN_Res101_20181204'
NET_NAME = 'resnet_v1_101'
ADD_BOX_IN_TENSORBOARD = True

# ---------------------------------------- System_config
ROOT_PATH = os.path.abspath('../')
print (20*"++--")
print (ROOT_PATH)
GPU_GROUP = "4"
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
#test_annotate_path = '/home/yjr/DataSet/VOC/VOC_test/VOC2007/Annotations'
test_annotate_path = '/home/gq123/dailinhui/FPN_success_dlh/data/pcb/Annotations'

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
#DECAY_STEP = [60000, 80000]  # 50000, 70000
DECAY_STEP = [1000, 1500]  # 50000, 70000
#MAX_ITERATION = 150000
MAX_ITERATION = 2000

# ------------------------------------------- Data_preprocess_config
DATASET_NAME = 'pcb'  # 'ship', 'spacenet', 'pascal', 'coco'
#PIXEL_MEAN = [123.68, 116.779, 103.939]  # R, G, B. In tf, channel is RGB. In openCV, channel is BGR
PIXEL_MEAN = [21.25, 85.936, 28.729]
IMG_SHORT_SIDE_LEN = 1586  # 600  # 600
IMG_MAX_LENGTH = 3034  # 1000  # 1000
CLASS_NUM = 6

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
SHOW_SCORE_THRSHOLD = 0.6  # only show in tensorboard

FAST_RCNN_NMS_IOU_THRESHOLD = 0.3  # 0.6
FAST_RCNN_NMS_MAX_BOXES_PER_CLASS = 100
FAST_RCNN_IOU_POSITIVE_THRESHOLD = 0.5
FAST_RCNN_IOU_NEGATIVE_THRESHOLD = 0.0   # 0.1 < IOU < 0.5 is negative
FAST_RCNN_MINIBATCH_SIZE = 256  # if is -1, that is train with OHEM
FAST_RCNN_POSITIVE_RATE = 0.25

ADD_GTBOXES_TO_TRAIN = False



