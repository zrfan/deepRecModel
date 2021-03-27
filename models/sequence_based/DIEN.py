# -*- coding: utf-8 -*
# tf1.14
import tensorflow as tf
import os
import sys
sys.path.append("../")
from models.base_estimator_model import BaseEstimatorModel
from models.ConfigParam import ConfigParam
from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.python.ops.rnn_cell import GRUCell

import random
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 使用第0块GPU
tf.set_random_seed(2020)
tf.reset_default_graph()

##
# Deep Interest Evolution Network for Click-Through Rate Prediction
##
class DIENModel(BaseEstimatorModel):
    def __init__(self, configParam):
        self.params = configParam
        self.data_path = configParam["data_path"]
        self.initialize_weights()
    def initialize_weights(self):
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

    def model_fn(self, features, labels, mode, params):
        pass
    def build_fcn_net(self, input, use_dice=False):
        bn1 = tf.layers.batch_normalization(inputs=input, name="bn1")
        dense_1 = tf.layers.dense(bn1, 200, activation=None, name="f1")
        if use_dice:
            dense_1 = self.dice(dense_1, name="dice_1")
        else :
            dense_1 = self.prelu(dense_1, name="prelu1")
        dense_2 = tf.layers.dense(dense_1, 80, activation=None, name="f2")
        if use_dice:
            dense_2 = self.dice(dense_2, name="dice_2")
        else:
            dense_2 = self.prelu(dense_2, name="prelu2")
        dense_3 = tf.layers.dense(dense_2, 2, activation=None, name="f3")
        y_hat = tf.nn.softmax(dense_3) + 0.00000001
        with tf.name_scope("Metrics"):
            ctr_loss = - tf.reduce_mean(tf.log(y_hat) * self.target_ph)
            self.loss = ctr_loss
            if self.params.use_negsampling:
                self.loss += self.aux_loss
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.params.learning_rate).minimize(self.loss)
            # Accuracy metric
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(y_hat), self.target), tf.float32))
    def auxiliary_loss(self, h_states, click_seq, noclick_seq, mask, stag=None):
        mask = tf.cast(mask, tf.float32)
        click_input = tf.concat([h_states, click_seq], -1)
        noclick_input = tf.concat([h_states, noclick_seq], -1)
        click_prop_ = self.auxiliary_net(click_input, stag=stag)[:, :, 0]
        noclick_prop = self.auxiliary_net(noclick_input, stag)[:, :, 0]
        click_loss = - tf.reshape(tf.log(click_prop_), [-1, tf.shape(click_seq)[1]]) * mask
        noclick_loss = - tf.reshape(tf.log(1.0 - noclick_prop), [-1, tf.shape(noclick_seq)[1]]) * mask
        loss = tf.reduce_mean(click_loss + noclick_loss)
        return loss
    def auxiliary_net(self, input, stag="auxiliary_net"):
        bn1 = tf.layers.batch_normalization(inputs=input, name="bn1"+stag, reuse=tf.AUTO_REUSE)
        dense_1 = tf.layers.dense(bn1, 100, activation=None, name="f1"+stag, reuse=tf.AUTO_REUSE)
        dense_1 = tf.nn.sigmoid(dense_1)
        dense_2 = tf.layers.dense(dense_1, 50, activation=None, name="f2"+stag, reuse=tf.AUTO_REUSE)
        dense_2 = tf.nn.sigmoid(dense_2)
        dense_3 = tf.layers.dense(dense_2, 2, activation=None, name="f3"+stag, reuse=tf.AUTO_REUSE)
        y_hat = tf.nn.softmax(dense_3) + 0.0000001
        return y_hat
    def dice(self, input, axis=-1, epsilon=0.000000001, name=""):
        alphas = tf.get_variable("alpha"+name, input.get_shape()[-1], initializeer=tf.constant_initializer(0.0), dtype=tf.float32)
        input_shape = list(input.get_shape())
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[axis] = input_shape[axis]

        # case: train mode (uses stats of the current batch)
        mean = tf.reduce_mean(input, axis=reduction_axes)
        broadcast_mean = tf.reshape(mean, broadcast_shape)
        std = tf.reduce_mean(tf.square(input - broadcast_mean)+epsilon, axis=reduction_axes)
        broadcast_std = tf.reshape(std, broadcast_shape)
        x_normed = (input - broadcast_mean) / (broadcast_std+epsilon)
        x_p = tf.sigmoid(x_normed)
        return alphas * (1.0 - x_p) * input + x_p * input
    def parametric_relu(self, input):
        alphas = tf.get_variable("alpha", input.get_shape()[-1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        pos = tf.nn.relu(input)
        neg = alphas * (input - abs(input)) * 0.5
        return neg + pos
    def din_fcn_attention(self, query, facts, attention_size, mask, stag="null", mode="sum", softmax_stag=1, time_major=False, return_alphas=False, forCnn=False):
        pass

class Model_Gru_att_Gru(DIENModel):
    def __init__(self, configParams):
        super(Model_Gru_att_Gru, self).__init__(configParams)
        ## RNN layer(-s)
        with tf.name_scope("rnn_1"):
            rnn_outputs, _ = dynamic_rnn(GRUCell(self.params.hidden_size), inputs=self.item_hist_emb,
                                         sequence_length=self.sequence_leng, dtype=tf.float32, scope="gru1")
        # Attention layer
        with tf.name_scope("attention_layer_1"):
            att_outputs, alphas = self.din_fcn_attention(self.item_emb, rnn_outputs, self.params.attention_size, self.mask,
                                                    softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True)
        with tf.name_scope("rnn_2"):
            rnn_outputs2, final_state2 = dynamic_rnn(GRUCell(self.params.hidden_size), inputs=att_outputs,
                                                     sequence_length=self.sequence_leng, dtype=tf.float32, scope="gru2")
        inp = tf.concat([self.uid_embedding, self.item_embedding, self.item_hist_embedding_sum, self.item_embedding*self.item_hist_embedding_sum, final_state2], 1)
        ## Fully connected layer
        self.build_fcn_net(inp, use_dice=True)
