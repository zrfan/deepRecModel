# -*- coding: utf-8 -*
# tf1.14
import tensorflow as tf
import os
from data_util import get1MTrainData

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 使用第0块GPU

class EssmParams(object):
    def __init__(self, params):
        self.params = params