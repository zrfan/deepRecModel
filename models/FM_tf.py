#-*- coding: utf-8 -*
# tf1.13
from __future__ import division
from math import exp
from numpy import *
from random import normalvariate  # 正态分布
from sklearn import preprocessing
import numpy as np
from .data_util import get1MTrainData
import tensorflow as tf

# https://zhuanlan.zhihu.com/p/145436595
class FMModelParams(object):
    """ class for initializing weights"""
    def __init__(self, data_path, feature_size, embedding_size=8):
        self.data_path, self.embedding_size = data_path, embedding_size
        self.feature_size = feature_size

    def initialize_weights(self):
        """ init fm weights
        returns
        weights: feature_embeddings: vi, vj second order params
                 weights_first_order: wi first order params
                 bias: b bias
        """
        weights = dict()
        weights_initializer = tf.glorot_normal_initializer()
        bias_initializer = tf.constant_initializer(0.0)
        weights["feature_embeddings"] = tf.get_variable(name="weights", dtype=tf.float32, initializer=weights_initializer,
                                                        shape=[self.feature_size, self.embedding_size])
        weights["weights_first_order"] = tf.get_variable(name="vector", dtype=tf.float32, initializer=weights_initializer,
                                                         shape=[self.feature_size, 1])
        weights["fm_bias"] = tf.get_variable(name="bias", dtype=tf.float32, initializer=bias_initializer, shape=[1])
        return weights
class FMModel(object):
    """ FM implementation for tensorflow"""
    @staticmethod
    def fm_model_fn(features, labels, mode, params):
        """ build tf model """
        embedding_size, feature_size, field_size = params["embedding_size"], params["feature_size"], params["field_size"]
        batch_size, learning_rate, optimizer_used = params["batch_size"], params["learning_rate"], params["optimizer"]
        feature_idx = features["feature_idx"]
        feature_idx = tf.reshape(feature_idx, shape=[batch_size, field_size])
        labels = tf.reshape(labels, shape=[batch_size, 1])
        feature_values = features["feature_values"]
        feature_values = tf.reshape(feature_values, shape=[batch_size, field_size])

        tf_model_params = FMModelParams(feature_size, embedding_size)
        weights = tf_model_params.initialize_weights()
        embeddings = tf.nn.embedding_lookup(weights["feature_embeddings"], feature_idx)
        weights_first_order = tf.nn.embedding_lookup(weights["weights_first_order"], feature_idx)
        bias = weights["fm_bias"]
        # build function
        ## first order
        first_order = tf.multiply(feature_values, weights_first_order)
        first_order = tf.reduce_sum(first_order, 2)
        first_order = tf.reduce_sum(first_order, 1, keepdims=True)

        ## second order
        ### feature * embedding
        feature_emb = tf.multiply(feature_values, embeddings)
        ### square(sum(feature * embedding))
        feature_emb_sum = tf.reduce_sum(feature_emb, 1)
        feature_emb_sum_square = tf.square(feature_emb_sum)
