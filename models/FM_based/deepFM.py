# -*- coding: utf-8 -*
# tf1.14
import tensorflow as tf
import os
import sys
import numpy as np
sys.path.append("../")
from models.data_util import get1MTrainDataOriginFeatures
from models.base_estimator_model import BaseEstimatorModel
from models.model_util import registerAllFeatureHashTable
from models.ConfigParam import ConfigParam
import random
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 使用第0块GPU
tf.set_random_seed(2019)
tf.reset_default_graph()


## https://github.com/ChenglongChen/tensorflow-DeepFM/blob/master/DeepFM.py
## https://zhuanlan.zhihu.com/p/48057256

class DeepFM(BaseEstimatorModel):
    def __init__(self, configParam):
        self.params = configParam
    def __initialize_weights(self):
        weights = dict()
        # embeddings feature_size * k
        weights["feature_embeddings"] = tf.Variable(tf.random_normal([self.params.feature_size, self.embedding_size],
                                                                                 0.0, 0.01), name="feature_embeddings")
        # feature_size * 1
        weights["feature_bias"] = tf.Variable(tf.random_uniform([self.params.feature_size, 1],
                                                                0.0, 1.0), name="feature_bias")
        # deep layers
        num_layers = len(self.params.deep_layers)
        input_size = self.params.field_size * self.params.embedding_size
        glorot = np.sqrt(2.0 / (input_size+self.params.deep_layers[0]))
        weights["layer_0"] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(input_size, self.params.deep_layers[0])),
                                         dtype=np.float32)
        weights["bias_0"] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.params.deep_layers[0])),
                                        dtype=np.float32)
        for i in range(1, num_layers):
            glorot = np.sqrt(2.0 / (self.params.deep_layers[i-1]+self.params.deep_layers[i]))
            # layers[i-1] * layers[i]
            weights["layer_%d"%i] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(self.params.deep_layers[i-1], self.params.deep_layers[i])),
                                                dtype=np.float32)
            # 1 * layers[i]
            weights["bias_%d"%i] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.params.deep_layers[i])),
                                               dtype=np.float32)