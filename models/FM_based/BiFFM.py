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
        self.seed = params["seed"]
    def initialize_weights(self):
        weights_initializer = tf.glorot_normal_initializer(seed=self.seed)
        bias_initializer = tf.constant_initializer(0.0)
        bias = tf.get_variable(name='bias', dtype=tf.float32, initializer=bias_initializer, shape=[1])
        first_order_weights = tf.get_variable(name="first_order_weights", shape=[self.feature_size, 1], dtype=tf.float32, initializer=weights_initializer)
        second_order_weights = tf.get_variable(name="second_order_weights",
                                               shape=[self.feature_size, self.embedding_size],
                                               dtype=tf.float32, initializer=weights_initializer)
        interaction_weights_shared = tf.get_variable(name="interaction_weights_shared", shape=[self.embedding_size, self.embedding_size],
                                              dtype=tf.float32, initializer=weights_initializer)
        interaction_weights_field = tf.get_variable(name="interaction_weights_field", shape=[self.field_size, self.embedding_size, self.embedding_size],
                                                    dtype=tf.float32, initializer=weights_initializer)
        return {"bias":bias, "first_order_weights": first_order_weights, "second_order_weights": second_order_weights,
                "interaction_weights_shared": interaction_weights_shared, "interaction_weights_field":interaction_weights_field}

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
        interaction_weights_shared = weights["interaction_weights_shared"]

        feature_idx, feature_values, feature_fields = feature["feature_idx"], feature["feature_values"], feature["feature_fields"]
        origin_features = feature["origin_features"]

        first_order = tf.multiply(tf.nn.embedding_lookup(first_order_weights, feature_idx), feature_values, name="first_order")
        first_order = tf.reduce_sum(tf.reduce_sum(first_order, 2), 1, keep_dims=True)

        feature_embeddings = tf.nn.embedding_lookup(all_embeddings, feature_idx)
        second_order = tf.multiply(feature_embeddings, interaction_weights_shared)





