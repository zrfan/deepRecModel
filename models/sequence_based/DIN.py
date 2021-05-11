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
tf.set_random_seed(2020)
tf.reset_default_graph()

##
# Deep Interest Network for Click-Through Rate Prediction
##
class DINModel(BaseEstimatorModel):
    def __init__(self, configParam):
        self.params = configParam
        self.data_path = configParam["data_path"]
    def init_weights(self):
        weights = dict()
        weights_initializer = tf.glorot_normal_initializer()
        bias_initializer = tf.constant_initializer(0.0)
        weights["user_embeddings"] = tf.get_variable(name="user_embeddings", dtype=tf.float32,
                                                        initializer=weights_initializer,
                                                        shape=[self.user_count + 1, self.params.embedding_size])
        weights["item_embeddings"] = tf.get_variable(name="item_embeddings", dtype=tf.float32,
                                                     initializer=weights_initializer,
                                                     shape=[self.item_count+1, self.params.embedding_size])
        weights["category_bias"] = tf.get_variable("category_bias", initializer=bias_initializer,
                                                   shape=[self.category_count])
        weights["category_emb"] = tf.get_variable("category_emb", initializer=weights_initializer,
                                                  shape=[self.category_count, self.params.embedding_size])
        weights["keyword_emb"] = tf.get_variable("keyword_emb", initializer=weights_initializer, shape=[self.keyword_count, self.params.embedding_size])
        weights["keyword2_emb"] = tf.get_variable("keyword2_emb", initializer=weights_initializer, shape=[self.keyword2_count, self.params.embedding_size])
        weights["tag1_emb"] = tf.get_variable("tag1_emb", initializer=weights_initializer, shape=[self.tag1_count, self.params.embedding_size])
        weights["tag2_emb"] = tf.get_variable("tag2_emb", initializer=weights_initializer, shape=[self.tag2_count, self.params.embedding_size])
        weights["tag3_emb"] = tf.get_variable("tag3_emb", initializer=weights_initializer, shape=[self.tag3_count, self.params.embedding_size])
        weights["ks1_emb"] = tf.get_variable("ks1_emb", initializer=weights_initializer, shape=[self.ks1_count, self.params.embedding_size])
        weights["ks2_emb"] = tf.get_variable("ks2_emb", initializer=weights_initializer, shape=[self.ks2_count, self.params.embedding_size])

    def model_fn(self, features, labels, mode, params):
        weights = self.init_weights()
        item_emb = tf.concat(values=[
            tf.nn.embedding_lookup(weights["category_emb"], features["category"]),
            tf.nn.embedding_lookup(weights["keyword_emb"], features["keyword"]),
            tf.nn.embedding_lookup(weights["keyword2_emb"], features["keyword2"]),
            tf.nn.embedding_lookup(weights["tag1_emb"], features["tag1"]),
            tf.nn.embedding_lookup(weights["tag2_emb"], features["tag2"]),
            tf.nn.embedding_lookup(weights["tag3_emb"], features["tag3"]),
            tf.nn.embedding_lookup(weights["ks1_emb"], features["ks1"]),
            tf.nn.embedding_lookup(weights["ks2_emb"], features["ks2"])
        ], axis=1)  # [Batch, embed_size*X]
        item_bias = tf.gather(weights["category_bias"], features["category"])
        history_emb = tf.concat(values=[
            tf.nn.embedding_lookup(weights["category_emb"], features["hist_category"]),
            tf.nn.embedding_lookup(weights["keyword_emb"], features["hist_keyword"]),
            tf.nn.embedding_lookup(weights["keyword2_emb"], features["hist_keyword2"]),
            tf.nn.embedding_lookup(weights["tag1_emb"], features["hist_tag1"]),
            tf.nn.embedding_lookup(weights["tag2_emb"], features["hist_tag2"]),
            tf.nn.embedding_lookup(weights["tag3_emb"], features["hist_tag3"]),
            tf.nn.embedding_lookup(weights["ks1_emb"], features["hist_ks1"]),
            tf.nn.embedding_lookup(weights["ks2_emb"], features["hist_ks2"])
        ], axis=2)  # [Batch, Time_len, embed_size*X]
        ## attention
        hist = self.attention(item_emb, history_emb, features["sequence_len"])

        # 普通Embedding层
        user_feature_emb = tf.nn.embedding_lookup(weights["user_embeddings"], features["user_feature"])  # [batch, embed_size, user_feature_num]
        user_feature_emb = tf.reshape(user_feature_emb, [-1, self.params.embedding_size*self.user_feature_num])
        item_feature_emb = tf.nn.embedding_lookup(weights["item_embeddings"], features["item_feature"])
        item_feature_emb = tf.reshape(item_feature_emb, [-1, self.params.embedding_size*self.item_feature_num])
        hist = tf.layers.batch_normalization(inputs=hist)
        hist = tf.reshape(hist, [-1, self.params.embedding_size])  # [batch, hidden_units]
        hist = tf.layers.dense(hist, self.params.embedding_size)
        user_emb = hist
        # 全连接层
        din_i = tf.concat([user_emb, user_feature_emb, item_feature_emb], axis=-1)
        din_i = tf.layers.batch_normalization(inputs=din_i, name="b0")

        # if you want try dice, change relu/sigmoid to None, and add dice layer like following
        # dense_1 = tf.layers.dense(din_i, self.params.deep_layers[0], activation=None, name="f1")
        # dense_1 = dice(dense_1, name="dice_1")
        dense_1 = tf.layers.dense(din_i, self.params.deep_layers[0], activation=tf.nn.relu, name="f1")
        dense_1 = tf.nn.dropout(dense_1, self.params.keep_prob)
        dense_1 = tf.layers.batch_normalization(inputs=dense_1, name="b1")

        dense_2 = tf.layers.dense(dense_1, self.params.deep_layers[1], activation=tf.nn.relu, name="f2")
        dense_2 = tf.nn.dropout(dense_2, self.params.keep_prob)
        dense_2 = tf.layers.batch_normalization(inputs=dense_2, name="b2")

        dense_3 = tf.layers.dense(dense_2, self.params.deep_layers[2], activation=tf.nn.relu, name="f3")
        dense_3 = tf.reshape(dense_3, [-1])  # 展开成行向量

        logits = item_bias + dense_3
        predicts = tf.sigmoid(logits, name="sig_logits")
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels["label"], logits=logits))
        tf.summary.scalar("loss", loss)
        optimizer = tf.train.AdamOptimizer(self.params.learning_rate, name="adam")
        train_op = optimizer.minimize(loss)

        eval_metric_ops = {"auc": tf.metrics.auc(labels["label"], predicts)}
        predictions = {"prob": predicts}

        export_outputs = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(predictions)}

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)
        elif mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=loss, eval_metric_ops=eval_metric_ops)
        elif mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=loss, eval_metric_ops=eval_metric_ops, train_op=train_op)
    def attention(self, queries, keys, keys_length):
        '''
        :param queries: [batch, embedding_size*X]       item_embedding
        :param keys:    [batch, Time, embedding_size*X]  history_embedding  #[B, T, H]
        :param keys_length: [batch] sequence_length
        :return:
        '''
        queries_hidden_units = queries.get_shape().as_list()[-1]  # queries_hidden_units = H
        queries = tf.tile(queries, [1, tf.shape(keys)[1]])  # [batch, H*T]
        queries = tf.reshape(queries, [-1, tf.shape(keys)[1], queries_hidden_units])  # [batch, Time, H]
        din_all = tf.concat([queries, keys, queries-keys, queries*keys], axis=-1)   # [batch, time, 4H]
        dense_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.relu, name="f1_att")  # [B, T, 80]
        dense_2_all = tf.layers.dense(dense_1_all,40,  activation=tf.nn.relu, name="f2_att")  # [B, T, 40]
        dense_3_all = tf.layers.dense(dense_2_all, 1, activation=None, name="f3_att")  # [B, T, 1]
        dense_3_all = tf.reshape(dense_3_all, [-1, 1, tf.shape(keys)[1]])  # [B, 1, T]

        outputs = dense_3_all
        # Mask
        key_mask = tf.sequence_mask(keys_length, tf.shape(keys)[1])  # [B, T]
        key_mask = tf.expand_dims(key_mask, 1)  # [B, 1, T]
        paddings = tf.ones_like(outputs)*(-2**32 + 1)
        outputs = tf.where(key_mask, outputs, paddings)  # [B, 1, T]

        # Scale
        outputs = outputs/(keys.get_shape().as_list()[-1]**0.5)
        # Activaton
        outputs = tf.nn.softmax(outputs)  # [B, 1, T]
        # Weighted sum
        outputs = tf.matmul(outputs, keys)  # [B, 1, H]
        return outputs
    def attention_multi_items(self, querys, keys, keys_length):
        '''
        :param querys:  [B, N, H] N is the number of ads
        :param keys:  [B, T, H]
        :param keys_length: [B]  sequence_length
        :return:
        '''
        querys_hidden_units = querys.get_shape().as_list()[-1]
        querys_num = querys.get_shape().as_list()[1]
        querys = tf.tile(querys, [1, 1, tf.shape(keys)[1]])
        querys = tf.reshape(querys, [-1, querys_num, tf.shape(keys)[1], querys_hidden_units])  # [B, N, T, H]
        max_len = tf.shape(keys)[1]
        keys = tf.tile(keys, [1, querys_num, 1])
        keys = tf.reshape(keys, [-1, querys_num, max_len, querys_hidden_units])  # [B, N, T, H]
        din_all = tf.concat([querys, keys, querys-keys, querys*keys], axis=-1)
        dense_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name="f1_att",  reuse=tf.AUTO_REUSE)
        dense_2_all = tf.layers.dense(dense_1_all, 40, activation=tf.nn.sigmoid, name="f2_att", reuse=tf.AUTO_REUSE)
        dense_3_all = tf.layers.dense(dense_2_all, 1, activation=None, name="f3_att",  reuse=tf.AUTO_REUSE)
        dense_3_all = tf.reshape(dense_3_all, [-1, querys_num, 1, max_len])  #[B, N, 1, T]
        outputs = dense_3_all
        # Mask
        key_mask = tf.sequence_mask(keys_length, max_len)  # [B, T]
        key_mask = tf.tile(key_mask, [1, querys_num])
        key_mask = tf.reshape(key_mask, [-1, querys_num, 1, max_len])  # [B, N, 1, T]
        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(key_mask, outputs, paddings)
        # Scale
        outputs = outputs / (keys.get_shape().as_list()[-1]**0.5)
        # Activation
        outputs = tf.nn.softmax(outputs)  # [B, N, 1, T]
        outputs = tf.reshape(outputs, [-1, 1, max_len])   #[B*N, 1, T]
        keys = tf.reshape(keys, [-1, max_len, querys_hidden_units])  # [B*N, T, H]
        # Weighted sum
        outputs = tf.matmul(outputs, keys)  # [B*N, 1, H]
        outputs = tf.reshape(outputs, [-1, querys_num, querys_hidden_units])  # [B, N, H]
        return outputs
    def dice(self, _x, axis=-1, epsilon=0.00000001, name=""):
        initializer = tf.constant_initializer(0.0)
        with tf.variable_scope(name_or_scope="", reuse=tf.AUTO_REUSE):
            alphas = tf.get_variable("alpha"+name, _x.get_shape()[-1], initializer=initializer, dtype=tf.float32)
            beta = tf.get_variable("beta"+name, _x.get_shape()[-1], initializer=initializer, dtype=tf.float32)
        input_shape = list(_x.get_shape())

        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[axis] = input_shape[axis]

        # case : train mode (uses stats of the current batch)
        mean = tf.reduce_mean(_x, axis=reduction_axes)
        broadcast_mean = tf.reshape(mean, broadcast_shape)
        std = tf.reduce_mean(tf.square(_x - broadcast_mean)+epsilon, axis=reduction_axes)
        std = tf.sqrt(std)
        broadcast_std = tf.reshape(std, broadcast_shape)
        x_normed = tf.layers.batch_normalization(_x, center=False, scale=False, name=name, reuse=tf.AUTO_REUSE)
        ## x_normed = (_x - broadcast_mean) / (broadcast_std + epsilon)
        x_p = tf.sigmoid(beta * x_normed)
        return alphas * (1.0 - x_p) * _x + x_p * _x
    def parametric_relu(self, _x):
        with tf.variable_scope(name_or_scope="", reuse=tf.AUTO_REUSE):
            alphas = tf.get_variable("alpha", _x.get_shape()[-1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        pos = tf.nn.relu(_x)
        neg = alphas * (_x - abs(_x)) * 0.5
        return pos+neg
class DINModel(BaseEstimatorModel):
    def __init__(self):
        pass
#     def train(self):
#         model_estimator = self.model_estimator(self.params)
#         # model.train(input_fn=self.train_input_fn, hooks=[tf.train.LoggingTensorHook(["input_layer", "ctr_score"], every_n_iter=500)])  ## input_layer
#         train_spec = tf.estimator.TrainSpec(input_fn=lambda : self.train_input_fn(f="train"), hooks=[tf.train.LoggingTensorHook(["dense_input", "expert_outputs", "ctr_score"], every_n_iter=500)])
#         eval_spec = tf.estimator.EvalSpec(input_fn=lambda : self.train_input_fn(f="test"), steps=None, start_delay_secs=1000, throttle_secs=1200)
#         tf.estimator.train_and_evaluate(model_estimator, train_spec, eval_spec)
#
# def main(_):
#     params = {"embedding_size": 6, "feature_size": 0, "field_size": 0, "batch_size": 64, "learning_rate": 0.001,"epochs":200,
#               "optimizer": "adam", "data_path": "../data/ml-1m/", "model_dir": "../data/model/mmoe/", "hidden_units":[8],
#               "experts_units": 2, "experts_num":2, "label1_weight": 0.5, "label2_weight": 0.5}
#     m = MMoEModel(configParam=ConfigParam(params))
#     m.train()
#
#
# if __name__ == '__main__':
#     print(tf.__version__)
#     tf.app.run()
#






