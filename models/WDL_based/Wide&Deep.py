# -*- coding: utf-8 -*
# tf1.14
import tensorflow as tf
import os
import sys
sys.path.append("../")
from models.data_util import get1MTrainDataOriginFeatures
from models.base_estimator_model import BaseEstimatorModel
from models.model_util import registerAllFeatureHashTable
from models.ConfigParam import ConfigParam
import random
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 使用第0块GPU
tf.set_random_seed(2019)
tf.reset_default_graph()

class WideDeep(BaseEstimatorModel):
    def __init__(self, configParam):
        self.params = configParam
        self.data_path = configParam["data_path"]
    def model_fn(self, features, labels, mode, params):

