# -*- coding: utf-8 -*-

import tensorflow.contrib.slim as slim
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""


@slim.add_arg_scope
def fn(a, c=100):

    return a+c


with slim.arg_scope([fn], a=2):
    with slim.arg_scope([fn], a=1):

        print(fn())


