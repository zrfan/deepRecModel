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
    def _initialize_weights(self):
        weights = dict()
        weights_initializer = tf.glorot_normal_initializer()
        bias_initializer = tf.constant_initializer(0.0)
        # embeddings feature_size * k
        weights["feature_embeddings"] = tf.get_variable(name="feature_embeddings", dtype=tf.float32,
                                                        initializer=weights_initializer,
                                                        shape=[self.params.feature_size, self.params.embedding_size])
        # feature_size * 1
        weights["feature_bias"] = tf.get_variable(name="feature_bias", dtype=tf.float32,
                                                  initializer=weights_initializer,
                                                  shape=[self.params.feature_size, 1])
        # deep layers
        num_layers = len(self.params.deep_layers)
        input_size = self.params.field_size * self.params.embedding_size
        weights["layer_0"] = tf.get_variable(name="layer_0", dtype=tf.float32,
                                             initializer=weights_initializer,
                                             shape=[input_size, self.params.deep_layers[0]])
        weights["bias_0"] = tf.get_variable(name="bias_0", dtype=tf.float32,
                                            initializer=weights_initializer,
                                            shape=[1, self.params.deep_layers[0]])
        for i in range(1, num_layers):
            # layers[i-1] * layers[i]
            weights["layer_%d"%i] = tf.get_variable(name="layer_%d"%i, dtype=tf.float32,
                                                    initializer=weights_initializer,
                                                    shape=[self.params.deep_layers[i-1], self.params.deep_layers[i]])
            # 1 * layers[i]
            weights["bias_%d"%i] = tf.get_variable(name="bias_%d"%i, dtype=tf.float32,
                                                   initializer=weights_initializer,
                                                   shape=[1, self.params.deep_layers[i]])
        # final concat projection layer
        if self.params.use_fm and self.params.use_deep:
            input_size = self.params.field_size + self.params.embedding_size + self.params.deep_layers[-1]
        elif self.params.use_fm:
            input_size = self.params.field_size + self.params.embedding_size
        elif self.params.use_deep:
            input_size = self.params.deep_layers[-1]
        weights["concat_projection"] = tf.get_variable(name="concat_projection", dtype=tf.float32,
                                                       initializer=weights_initializer,
                                                       shape=[input_size, 1])
        weights["concat_bias"] = tf.get_variable(name="concat_bias", dtype=tf.float32,
                                                 initializer=bias_initializer, shape=[1])
        return weights
    def deepFM_model_fn(self, features, labels, mode):
        """build tf model"""
        batch_size, learning_rate, optimizer_used = self.params.batch_size, self.params.learning_rate, self.params.optimizer_used
        feature_idx, feature_values = features["feature_idx"], features["feature_values"]
        feature_idx = tf.reshape(feature_idx, shape=[batch_size, tf.shape(feature_idx)[1], 1])
        feature_values = tf.reshape(feature_values, shape=[batch_size, tf.shape(feature_values)[1], 1])
        labels = tf.reshape(labels, shape=[batch_size, 1])
        weights = self._initialize_weights()
        embeddings = tf.nn.embedding_lookup(weights["feature_embeddings"], feature_idx)

        #### ---------------- first order term -------------
        feature_bias = tf.nn.embedding_lookup(weights["feature_bias"], feature_idx)
        first_order = tf.multiply(feature_bias, feature_values, name="first_order")
        first_order = tf.reduce_sum(first_order, 2)
        ###### dropout
        first_order = tf.nn.dropout(first_order, self.params.dropout_keep_fm[0])
        # first_order = tf.reduce_sum(first_order, 1, keepdims=True)

        #### ---------------- second order term ------------
        ###### feature * embeddings
        embeddings = tf.multiply(embeddings, feature_values)
        ###### square_sum part    square(sum(feature * embedding))
        feature_emb_sum = tf.reduce_sum(embeddings, 1)
        feature_emb_sum_square = tf.square(feature_emb_sum)

        ###### sum_square part     sum(square(feature * embedding))
        feature_emb_square = tf.square(embeddings)
        feature_emb_square_sum = tf.reduce_sum(feature_emb_square, 1)

        second_order = feature_emb_sum_square - feature_emb_square_sum
        ####### dropout
        second_order = tf.nn.dropout(second_order, self.params.dropout_keep_fm[1])
        # second_order = tf.reduce_sum(second_order, axis=1, keepdims=True)

        ##### -------------- Deep component  -----------
        deep = tf.reshape(embeddings, shape=[-1, self.params.field_size * self.params.embedding_size])
        deep = tf.nn.dropout(deep, self.params.dropout_keep_deep[0])
        for i in range(0, len(self.params.deep_layers)):
            deep = tf.matmul(deep, weights["layer_%d"%i]) + weights["bias_%d"%i]
            if self.params.batch_norm:
                deep = self.batch_norm_layer(deep, train_phase=self.params.train_phase, scope_bn="bn_%d"%i)
            deep = self.deep_layers_activation(deep)
            deep = tf.nn.dropout(deep, self.params.dropout_keep_deep[i+1])
        ##### -------------- DeepFM -------------
        if self.params.use_fm and self.params.use_deep:
            concat_input = tf.concat([first_order, second_order, deep], axis=1)
        elif self.params.use_fm:
            concat_input = tf.concat([first_order, second_order], axis=1)
        elif self.params.use_deep:
            concat_input = deep
        predicts = tf.matmul(concat_input, weights["concat_projection"]) + weights["concat_bias"]
        #### loss
        if self.params.loss_type == "logloss":
            predicts = tf.nn.sigmoid(predicts)
            loss = tf.losses.log_loss(labels, predicts)
        elif self.params.loss_type == "mse":
            loss = tf.nn.l2_loss(labels - predicts)
        ### l2 regularization on weights
        if self.params.l2_reg > 0:
            loss += tf.contrib.layers.l2_regularizer(self.params.l2_reg)(weights["concat_projection"])
            if self.params.use_deep:
                for i in range(len(self.params.deep_layers)):
                    loss += tf.contrib.layers.l2_regularizer(self.params.l2_reg)(weights["layer_%d"%i])
        #### optimizer
        if optimizer_used == "adam":
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        elif optimizer_used == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
        elif optimizer_used == "sgd":
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        elif optimizer_used == "momentum":
            optimizer = tf.train.MomentumOptimizer(learning_rate)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        # metric
        eval_metric_ops = {"auc": tf.metrics.auc(labels, predicts)}
        predictions = {"prob": predicts}

        # return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.TRAIN, predictions=predicts, loss=loss,
        #                                   eval_metric_ops=eval_metric_ops, train_op=train_op)
        export_outputs = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(predictions)}
        # predict 输出
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)
        elif mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=loss, eval_metric_ops=eval_metric_ops)
        elif mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=loss, eval_metric_ops=eval_metric_ops, train_op=train_op)





