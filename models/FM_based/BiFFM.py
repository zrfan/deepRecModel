# -*- coding: utf-8 -*
# tf1.14
import numpy as np
import pandas as pd
import tensorflow as tf
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 使用第0块GPU

class BiFFMParams(object):
    def __init__(self, params):
        self.embedding_size, self.field_size, self.feature_size = params["embedding_size"], params["field_size"], params["feature_size"]
    def initialize_weights(self):
        weights_initializer = tf.glorot_normal_initializer()
        bias_initializer = tf.constant_initializer(0.0)
        bias = tf.get_variable(name='bias', dtype=tf.float32, initializer=bias_initializer, shape=[1])
        first_order_weights = tf.get_variable(name="first_order_weights", shape=[self.feature_size, 1], dtype=tf.float32, initializer=weights_initializer)
        second_order_weights = tf.get_variable(name="second_order_weights",
                                               shape=[self.feature_size, self.embedding_size],
                                               dtype=tf.float32, initializer=weights_initializer)
        return {"bias":bias, "first_order_weights": first_order_weights, "second_order_weights": second_order_weights}

class BiFFMModel(object):
    """Bilinear FFM implementation of tensorflow"""
    def __init__(self, data_path, params):
        self.data_path, self.params = data_path, params
    def biffm_model_fn(self, feature, labels, mode):
        batch_size, learning_rate, optimizer_used = self.params["batch_size"], self.params["learning_rate"], self.params["optimizer_used"]

        weights = BiFFMParams(params=self.params)
        all_embeddings = weights["second_order_weights"]
        first_order_weights = weights["first_order_weights"]
        bias = weights["bias"]



