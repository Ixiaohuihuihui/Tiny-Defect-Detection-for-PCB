# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, division


import tensorflow as tf
import tensorflow.contrib.slim as slim
from libs.configs import cfgs
from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.contrib.slim.nets import resnet_utils
from tensorflow.contrib.slim.python.slim.nets.resnet_v1 import resnet_v1_block
import tfplot as tfp


def resnet_arg_scope(
        is_training=True, weight_decay=cfgs.WEIGHT_DECAY, batch_norm_decay=0.997,
        batch_norm_epsilon=1e-5, batch_norm_scale=True):
    '''

    In Default, we do not use BN to train resnet, since batch_size is too small.
    So is_training is False and trainable is False in the batch_norm params.

    '''
    batch_norm_params = {
        'is_training': False, 'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon, 'scale': batch_norm_scale,
        'trainable': False,
        'updates_collections': tf.GraphKeys.UPDATE_OPS
    }

    with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.variance_scaling_initializer(),
            trainable=is_training,
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
            return arg_sc


def fusion_two_layer(C_i, P_j, scope):
    '''
    i = j+1
    :param C_i: shape is [1, h, w, c]
    :param P_j: shape is [1, h/2, w/2, 256]
    :return:
    P_i
    '''
    with tf.variable_scope(scope):
        level_name = scope.split('_')[1]
        h, w = tf.shape(C_i)[1], tf.shape(C_i)[2]
        upsample_p = tf.image.resize_bilinear(P_j,
                                              size=[h, w],
                                              name='up_sample_'+level_name)

        reduce_dim_c = slim.conv2d(C_i,
                                   num_outputs=256,
                                   kernel_size=[1, 1], stride=1,
                                   scope='reduce_dim_'+level_name)

        add_f = 0.5*upsample_p + 0.5*reduce_dim_c

        # P_i = slim.conv2d(add_f,
        #                   num_outputs=256, kernel_size=[3, 3], stride=1,
        #                   padding='SAME',
        #                   scope='fusion_'+level_name)
        return add_f


def add_heatmap(feature_maps, name):
    '''

    :param feature_maps:[B, H, W, C]
    :return:
    '''

    def figure_attention(activation):
        fig, ax = tfp.subplots()
        im = ax.imshow(activation, cmap='jet')
        fig.colorbar(im)
        return fig

    heatmap = tf.reduce_sum(feature_maps, axis=-1)
    heatmap = tf.squeeze(heatmap, axis=0)
    tfp.summary.plot(name, figure_attention, [heatmap])


def resnet_base(img_batch, scope_name, is_training=True):
    '''
    this code is derived from light-head rcnn.
    https://github.com/zengarden/light_head_rcnn

    It is convenient to freeze blocks. So we adapt this mode.
    '''
    if scope_name == 'resnet_v1_50':
        middle_num_units = 6
    elif scope_name == 'resnet_v1_101':
        middle_num_units = 23
    else:
        raise NotImplementedError('We only support resnet_v1_50 or resnet_v1_101. Check your network name....yjr')

    blocks = [resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
              resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
              resnet_v1_block('block3', base_depth=256, num_units=middle_num_units, stride=2),
              resnet_v1_block('block4', base_depth=512, num_units=3, stride=1)]
    # when use fpn . stride list is [1, 2, 2]

    with slim.arg_scope(resnet_arg_scope(is_training=False)):
        with tf.variable_scope(scope_name, scope_name):
            # Do the first few layers manually, because 'SAME' padding can behave inconsistently
            # for images of different sizes: sometimes 0, sometimes 1
            net = resnet_utils.conv2d_same(
                img_batch, 64, 7, stride=2, scope='conv1')
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
            net = slim.max_pool2d(
                net, [3, 3], stride=2, padding='VALID', scope='pool1')

    not_freezed = [False] * cfgs.FIXED_BLOCKS + (4-cfgs.FIXED_BLOCKS)*[True]
    # Fixed_Blocks can be 1~3

    with slim.arg_scope(resnet_arg_scope(is_training=(is_training and not_freezed[0]))):
        C2, end_points_C2 = resnet_v1.resnet_v1(net,
                                                blocks[0:1],
                                                global_pool=False,
                                                include_root_block=False,
                                                scope=scope_name)

    # C2 = tf.Print(C2, [tf.shape(C2)], summarize=10, message='C2_shape')
    add_heatmap(C2, name='Layer2/C2_heat')

    with slim.arg_scope(resnet_arg_scope(is_training=(is_training and not_freezed[1]))):
        C3, end_points_C3 = resnet_v1.resnet_v1(C2,
                                                blocks[1:2],
                                                global_pool=False,
                                                include_root_block=False,
                                                scope=scope_name)

    # C3 = tf.Print(C3, [tf.shape(C3)], summarize=10, message='C3_shape')
    add_heatmap(C3, name='Layer3/C3_heat')
    with slim.arg_scope(resnet_arg_scope(is_training=(is_training and not_freezed[2]))):
        C4, end_points_C4 = resnet_v1.resnet_v1(C3,
                                                blocks[2:3],
                                                global_pool=False,
                                                include_root_block=False,
                                                scope=scope_name)

    add_heatmap(C4, name='Layer4/C4_heat')

    # C4 = tf.Print(C4, [tf.shape(C4)], summarize=10, message='C4_shape')
    with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
        C5, end_points_C5 = resnet_v1.resnet_v1(C4,
                                                blocks[3:4],
                                                global_pool=False,
                                                include_root_block=False,
                                                scope=scope_name)
    # C5 = tf.Print(C5, [tf.shape(C5)], summarize=10, message='C5_shape')
    add_heatmap(C5, name='Layer5/C5_heat')

    feature_dict = {'C2': end_points_C2['{}/block1/unit_2/bottleneck_v1'.format(scope_name)],
                    'C3': end_points_C3['{}/block2/unit_3/bottleneck_v1'.format(scope_name)],
                    'C4': end_points_C4['{}/block3/unit_{}/bottleneck_v1'.format(scope_name, middle_num_units - 1)],
                    'C5': end_points_C5['{}/block4/unit_3/bottleneck_v1'.format(scope_name)],
                    # 'C5': end_points_C5['{}/block4'.format(scope_name)],
                    }

    # feature_dict = {'C2': C2,
    #                 'C3': C3,
    #                 'C4': C4,
    #                 'C5': C5}

    pyramid_dict = {}
    with tf.variable_scope('build_pyramid'):
        with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(cfgs.WEIGHT_DECAY),
                            activation_fn=None, normalizer_fn=None):

            P5 = slim.conv2d(C5,
                             num_outputs=256,
                             kernel_size=[1, 1],
                             stride=1, scope='build_P5')
            if "P6" in cfgs.LEVLES:
                P6 = slim.max_pool2d(P5, kernel_size=[1, 1], stride=2, scope='build_P6')
                pyramid_dict['P6'] = P6

            pyramid_dict['P5'] = P5

            for level in range(4, 1, -1):  # build [P4, P3, P2]

                pyramid_dict['P%d' % level] = fusion_two_layer(C_i=feature_dict["C%d" % level],
                                                               P_j=pyramid_dict["P%d" % (level+1)],
                                                               scope='build_P%d' % level)
            for level in range(4, 1, -1):
                pyramid_dict['P%d' % level] = slim.conv2d(pyramid_dict['P%d' % level],
                                                          num_outputs=256, kernel_size=[3, 3], padding="SAME",
                                                          stride=1, scope="fuse_P%d" % level)
    for level in range(5, 1, -1):
        add_heatmap(pyramid_dict['P%d' % level], name='Layer%d/P%d_heat' % (level, level))

    # return [P2, P3, P4, P5, P6]
    print("we are in Pyramid::-======>>>>")
    print(cfgs.LEVLES)
    print("base_anchor_size are: ", cfgs.BASE_ANCHOR_SIZE_LIST)
    print(20 * "__")
    return [pyramid_dict[level_name] for level_name in cfgs.LEVLES]
    # return pyramid_dict  # return the dict. And get each level by key. But ensure the levels are consitant
    # return list rather than dict, to avoid dict is unordered



def restnet_head(inputs, is_training, scope_name):
    '''

    :param inputs: [minibatch_size, 7, 7, 256]
    :param is_training:
    :param scope_name:
    :return:
    '''

    with tf.variable_scope('build_fc_layers'):

        # fc1 = slim.conv2d(inputs=inputs,
        #                   num_outputs=1024,
        #                   kernel_size=[7, 7],
        #                   padding='VALID',
        #                   scope='fc1') # shape is [minibatch_size, 1, 1, 1024]
        # fc1 = tf.squeeze(fc1, [1, 2], name='squeeze_fc1')

        inputs = slim.flatten(inputs=inputs, scope='flatten_inputs')

        fc1 = slim.fully_connected(inputs, num_outputs=1024, scope='fc1')

        fc2 = slim.fully_connected(fc1, num_outputs=1024, scope='fc2')

        # fc3 = slim.fully_connected(fc2, num_outputs=1024, scope='fc3')

        # we add fc3 to increase the ability of fast-rcnn head
        return fc2






























