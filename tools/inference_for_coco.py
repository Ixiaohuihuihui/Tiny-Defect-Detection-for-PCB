# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os, sys
import tensorflow as tf
import time
import cv2
import pickle
import numpy as np
sys.path.append("../")
sys.path.insert(0, '/home/yjr/PycharmProjects/Faster-RCNN_TF/data/lib_coco/PythonAPI')
from data.io.image_preprocess import short_side_resize_for_inference_data
from libs.configs import cfgs
from libs.networks import build_whole_network
from libs.val_libs import voc_eval
from libs.box_utils import draw_box_in_img
from libs.label_name_dict.coco_dict import LABEL_NAME_MAP, classes_originID
from help_utils import tools
from data.lib_coco.PythonAPI.pycocotools.coco import COCO
import json

os.environ["CUDA_VISIBLE_DEVICES"] = cfgs.GPU_GROUP


def eval_with_plac(det_net, imgId_list, coco, out_json_root, draw_imgs=False):

    # 1. preprocess img
    img_plac = tf.placeholder(dtype=tf.uint8, shape=[None, None, 3])  # is RGB. not GBR
    img_batch = tf.cast(img_plac, tf.float32)

    img_batch = short_side_resize_for_inference_data(img_tensor=img_batch,
                                                     target_shortside_len=cfgs.IMG_SHORT_SIDE_LEN,
                                                     length_limitation=cfgs.IMG_MAX_LENGTH)
    img_batch = img_batch - tf.constant(cfgs.PIXEL_MEAN)
    img_batch = tf.expand_dims(img_batch, axis=0)

    detection_boxes, detection_scores, detection_category = det_net.build_whole_detection_network(
        input_img_batch=img_batch,
        gtboxes_batch=None)

    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )

    restorer, restore_ckpt = det_net.get_restorer()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # coco_test_results = []

    with tf.Session(config=config) as sess:
        sess.run(init_op)
        if not restorer is None:
            restorer.restore(sess, restore_ckpt)
            print('restore model')

        for i, imgid in enumerate(imgId_list):
            imgname = coco.loadImgs(ids=[imgid])[0]['file_name']
            raw_img = cv2.imread(os.path.join("/home/yjr/DataSet/COCO/2017/test2017", imgname))

            raw_h, raw_w = raw_img.shape[0], raw_img.shape[1]
            start = time.time()
            resized_img, detected_boxes, detected_scores, detected_categories = \
                sess.run(
                    [img_batch, detection_boxes, detection_scores, detection_category],
                    feed_dict={img_plac: raw_img[:, :, ::-1]}  # cv is BGR. But need RGB
                )
            end = time.time()

            if draw_imgs:
                show_indices = detected_scores >= cfgs.SHOW_SCORE_THRSHOLD
                show_scores = detected_scores[show_indices]
                show_boxes = detected_boxes[show_indices]
                show_categories = detected_categories[show_indices]
                final_detections = draw_box_in_img.draw_boxes_with_label_and_scores(np.squeeze(resized_img, 0),
                                                                                    boxes=show_boxes,
                                                                                    labels=show_categories,
                                                                                    scores=show_scores)
                cv2.imwrite(cfgs.TEST_SAVE_PATH + '/' + str(imgid) + '.jpg',
                            final_detections[:, :, ::-1])

            xmin, ymin, xmax, ymax = detected_boxes[:, 0], detected_boxes[:, 1], \
                                     detected_boxes[:, 2], detected_boxes[:, 3]

            resized_h, resized_w = resized_img.shape[1], resized_img.shape[2]

            xmin = xmin * raw_w / resized_w
            xmax = xmax * raw_w / resized_w

            ymin = ymin * raw_h / resized_h
            ymax = ymax * raw_h / resized_h

            boxes = np.transpose(np.stack([xmin, ymin, xmax-xmin, ymax-ymin]))

            dets = np.hstack((detected_categories.reshape(-1, 1),
                              detected_scores.reshape(-1, 1),
                              boxes))

            a_img_detect_result = []
            for a_det in dets:
                label, score, bbox = a_det[0], a_det[1], a_det[2:]
                cat_id = classes_originID[LABEL_NAME_MAP[label]]
                if score<0.00001:
                   continue
                det_object = {"image_id": imgid,
                              "category_id": cat_id,
                              "bbox": bbox.tolist(),
                              "score": float(score)}
                # print (det_object)
                a_img_detect_result.append(det_object)
            f = open(os.path.join(out_json_root, 'each_img', str(imgid)+'.json'), 'w')
            json.dump(a_img_detect_result, f)  # , indent=4
            f.close()
            del a_img_detect_result
            del dets
            del boxes
            del resized_img
            del raw_img
            tools.view_bar('{} image cost {}s'.format(imgid, (end - start)), i + 1, len(imgId_list))


def eval(num_imgs):


   # annotation_path = '/home/yjr/DataSet/COCO/2017/test_annotations/image_info_test2017.json'
    annotation_path = '/home/yjr/DataSet/COCO/2017/test_annotations/image_info_test-dev2017.json'
    # annotation_path = '/home/yjr/DataSet/COCO/2017/annotations/instances_train2017.json'
    print("load coco .... it will cost about 17s..")
    coco = COCO(annotation_path)

    imgId_list = coco.getImgIds()

    if num_imgs !=np.inf:
        imgId_list = imgId_list[: num_imgs]

    faster_rcnn = build_whole_network.DetectionNetwork(base_network_name=cfgs.NET_NAME,
                                                       is_training=False)
    save_dir = os.path.join(cfgs.EVALUATE_DIR, cfgs.VERSION)
    eval_with_plac(det_net=faster_rcnn, coco=coco, imgId_list=imgId_list, out_json_root=save_dir,
                   draw_imgs=True)
    print("each img over**************")

    final_detections = []
    with open(os.path.join(save_dir, 'coco2017test_results.json'), 'w') as wf:
        for imgid in imgId_list:
            f = open(os.path.join(save_dir, 'each_img', str(imgid)+'.json'))
            tmp_list = json.load(f)
            # print (type(tmp_list))
            final_detections.extend(tmp_list)
            del tmp_list
            f.close()
        json.dump(final_detections, wf)


if __name__ == '__main__':

    eval(np.inf)

















