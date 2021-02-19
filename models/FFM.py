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
        weights["feature_embeddings"] = tf.get_variable(name="weights", dtype=tf.float32, initializer=weights_initializer,
                                                       shape=[self.feature_size+1, self.field_size, self.embedding_size])
        weights["weight_first_order"] = tf.get_variable(name="first_vector", dtype=tf.float32, initializer=weights_initializer,
                                                        shape=[self.feature_size+1, 1])
        weights["ffm_bias"] = tf.get_variable(name="bias", dtype=tf.float32, initializer=bias_initializer, shape=[1])
        return weights

# https://zhuanlan.zhihu.com/p/145928996
# https://github.com/wziji/deep_ctr
class FFMModel(object):
    """Field-aware of Factorization Machine implementation for tensorflow"""
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
        # feature_fields = tf.reshape(feature_fields, shape=[batch_size, tf.shape(feature_fields)[1], 1])

        ffm_params = FFMParams(self.params)
        all_weights = ffm_params.initialize_weights()
        bias = all_weights["bias"]
        all_embedding = all_weights["feature_embeddings"]
        weights_first_order = tf.nn.embedding_lookup(all_weights["weights_first_order"], feature_idx)


        ## first_order
        first_order = tf.multiply(weights_first_order, feature_values, name="first_order")
        first_order = tf.reduce_sum(first_order, 2)
        first_order = tf.reduce_sum(first_order, 1, keep_dims=True)

        ## second_order
        second_order = tf.constant(0, dtype=tf.float32)
        input_number = tf.shape(feature_idx).as_list()[1]
        for i in range(input_number):
            for j in range(i+1, input_number):
                idx_i, idx_j = feature_idx[:, i, :], feature_idx[:, j, :]
                field_i, field_j = feature_fields[:, i], feature_fields[:, j]
                emb_i, emb_j = all_embedding[idx_i, field_j, :], all_embedding[idx_j, field_i, :]
                val_i, val_j = feature_values[:, i, :], feature_values[:, j, :]

                field_emb_sum = tf.multiply(emb_i, emb_j)
                val_sum = tf.multiply(val_i, val_j)

                sum = tf.multiply(tf.reduce_sum(field_emb_sum, axis=1), val_sum)
                second_order += sum
        ## final objective function
        logits = second_order + first_order + bias
        predicts = tf.sigmoid(logits)

        ## loss function
        sigmoid_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
        sigmoid_loss = tf.reduce_mean(sigmoid_loss, name="sigmoid_loss")
        loss = sigmoid_loss

        # train_op
        if optimizer_used == "adagrad":
            optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
        elif optimizer_used == "adam":
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        else:
            raise Exception("unknown optimizer", optimizer_used)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        # metric
        eval_metric_ops = {"auc": tf.metrics.auc(labels, predicts)}
        predictions = {"prob": predicts}
        return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.TRAIN, predictions=predicts, loss=loss,
                                          eval_metric_ops=eval_metric_ops, train_op=train_op)



