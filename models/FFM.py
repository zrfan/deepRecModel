# -*- coding: utf-8 -*
# tf1.14
import pandas
import tensorflow as tf
import os
import numpy

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 使用第0块GPU

class FFMParams(object):
    def __init__(self, params):
        self.embedding_size, self.feature_size = params["embedding_size"], params["feature_size"]
        self.field_size = params["field_size"]
    def initialize_weights(self):
        weights = dict()
        weights_initializer = tf.glorot_normal_initializer()
        bias_initializer = tf.constant_initializer(0.0)
        weights["feature_embedding"] = tf.get_variable(name="weights", dtype=tf.float32, initializer=weights_initializer,
                                                       shape=[self.feature_size+1, self.field_size, self.embedding_size])
        weights["weight_first_order"] = tf.get_variable(name="first_vector", dtype=tf.float32, initializer=weights_initializer,
                                                        shape=[self.feature_size+1, 1])
        weights["ffm_bias"] = tf.get_variable(name="bias", dtype=tf.float32, initializer=bias_initializer, shape=[1])
        return weights

# https://zhuanlan.zhihu.com/p/145928996
# https://github.com/wziji/deep_ctr
class FFMModel(object):
    def __init__(self, data_path, params):
        self.data_path, self.params = data_path, params
    def ffm_model_fn(self, features, labels, mode):
        embedding_size, feature_size, field_size = self.params["embedding_size"], self.params["feature_size"], self.params["field_size"]
        batch_size, learning_rate, optimizer_used = self.params["batch_size"], self.params["learning_rate"], self.params["optimizer"]
        feature_idx = features["feature_idx"]
        feature_idx = tf.reshape(feature_idx, shape=[batch_size, tf.shape(feature_idx)[1]])
        labels = tf.reshape(labels, shape=[batch_size, 1])
        feature_values = features["feature_values"]
        feature_values = tf.reshape(feature_values, shape=[batch_size, tf.shape(feature_values)[1], 1])
        feature_fields = features["feature_fields"]
        feature_fields = tf.reshape(feature_fields, shape=[batch_size, tf.shape(feature_fields)[1], 1])

        ffm_params = FFMParams(self.params)
        all_weights = ffm_params.initialize_weights()
        all_embedding = all_weights["feature_embeddi"]

