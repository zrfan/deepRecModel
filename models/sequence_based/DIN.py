# -*- coding: utf-8 -*
# tf1.14
import tensorflow as tf
import os
import sys
sys.path.append("../")
from models.base_estimator_model import BaseEstimatorModel
from models.ConfigParam import ConfigParam

import random
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 使用第0块GPU
tf.set_random_seed(2019)
tf.reset_default_graph()

class DINModel(BaseEstimatorModel):
    def __init__(self, configParam):
        self.params = configParam
        self.data_path = configParam["data_path"]
    def model_fn(self, features, labels, mode, params):
