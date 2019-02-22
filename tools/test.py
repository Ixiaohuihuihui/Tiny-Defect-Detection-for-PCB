# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os, sys
import tensorflow as tf
import time
import cv2
import argparse
import numpy as np
sys.path.append("../")

from data.io.image_preprocess import short_side_resize_for_inference_data
from libs.configs import cfgs
from libs.networks import build_whole_network
from libs.box_utils import draw_box_in_img
from help_utils import tools


def detect(det_net, inference_save_path, real_test_imgname_list):

    # 1. preprocess img
    img_plac = tf.placeholder(dtype=tf.uint8, shape=[None, None, 3])  # is RGB. not GBR
    img_batch = tf.cast(img_plac, tf.float32)
    img_batch = short_side_resize_for_inference_data(img_tensor=img_batch,
                                                     target_shortside_len=cfgs.IMG_SHORT_SIDE_LEN,
                                                     length_limitation=cfgs.IMG_MAX_LENGTH)
    img_batch = img_batch - tf.constant(cfgs.PIXEL_MEAN)
    img_batch = tf.expand_dims(img_batch, axis=0) # [1, None, None, 3]

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

    with tf.Session(config=config) as sess:
        sess.run(init_op)
        if not restorer is None:
            restorer.restore(sess, restore_ckpt)
            print('restore model')

        for i, a_img_name in enumerate(real_test_imgname_list):

            raw_img = cv2.imread(a_img_name)
            start = time.time()
            resized_img, detected_boxes, detected_scores, detected_categories = \
                sess.run(
                    [img_batch, detection_boxes, detection_scores, detection_category],
                    feed_dict={img_plac: raw_img[:, :, ::-1]}  # cv is BGR. But need RGB
                )
            end = time.time()
            # print("{} cost time : {} ".format(img_name, (end - start)))

            raw_h, raw_w = raw_img.shape[0], raw_img.shape[1]

            xmin, ymin, xmax, ymax = detected_boxes[:, 0], detected_boxes[:, 1], \
                                     detected_boxes[:, 2], detected_boxes[:, 3]

            resized_h, resized_w = resized_img.shape[1], resized_img.shape[2]

            xmin = xmin * raw_w / resized_w
            xmax = xmax * raw_w / resized_w

            ymin = ymin * raw_h / resized_h
            ymax = ymax * raw_h / resized_h

            detected_boxes = np.transpose(np.stack([xmin, ymin, xmax, ymax]))

            show_indices = detected_scores >= cfgs.SHOW_SCORE_THRSHOLD
            show_scores = detected_scores[show_indices]
            show_boxes = detected_boxes[show_indices]
            show_categories = detected_categories[show_indices]
            final_detections = draw_box_in_img.draw_boxes_with_label_and_scores(raw_img - np.array(cfgs.PIXEL_MEAN),
                                                                                boxes=show_boxes,
                                                                                labels=show_categories,
                                                                                scores=show_scores)
            nake_name = a_img_name.split('/')[-1]
            # print (inference_save_path + '/' + nake_name)
            cv2.imwrite(inference_save_path + '/' + nake_name,
                        final_detections[:, :, ::-1])

            tools.view_bar('{} image cost {}s'.format(a_img_name, (end - start)), i + 1, len(real_test_imgname_list))


def test(test_dir, inference_save_path):

    test_imgname_list = [os.path.join(test_dir, img_name) for img_name in os.listdir(test_dir)
                                                          if img_name.endswith(('.jpg', '.png', '.jpeg', '.tif', '.tiff'))]
    assert len(test_imgname_list) != 0, 'test_dir has no imgs there.' \
                                        ' Note that, we only support img format of (.jpg, .png, and .tiff) '

    faster_rcnn = build_whole_network.DetectionNetwork(base_network_name=cfgs.NET_NAME,
                                                       is_training=False)
    detect(det_net=faster_rcnn, inference_save_path=inference_save_path, real_test_imgname_list=test_imgname_list)


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='TestImgs...U need provide the test dir')
    parser.add_argument('--data_dir', dest='data_dir',
                        help='data path',
                        default='demos', type=str)
    parser.add_argument('--save_dir', dest='save_dir',
                        help='demo imgs to save',
                        default='inference_results', type=str)
    parser.add_argument('--GPU', dest='GPU',
                        help='gpu id ',
                        default='0', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    return args
if __name__ == '__main__':

    args = parse_args()
    print('Called with args:')
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
    test(args.data_dir,
         inference_save_path=args.save_dir)
















