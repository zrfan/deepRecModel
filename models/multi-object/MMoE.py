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

##
# Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts
##
class MMoEModel(BaseEstimatorModel):
    def __init__(self, configParam):
        self.params = configParam
        print("get params=", self.params)
    def model_fn(self, features, labels, mode, params):
        gender_column = tf.feature_column.categorical_column_with_vocabulary_list("gender", ['M', 'F'])
        gender_column = tf.feature_column.embedding_column(gender_column, 2)
        print("***************gender=", gender_column)
        age_column = tf.feature_column.categorical_column_with_vocabulary_list("age", self.ageList)
        age_column = tf.feature_column.embedding_column(age_column, 2)
        # occupationList = ['0', '1', '2', '3', '4', '5', '6', 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        occupation_column = tf.feature_column.categorical_column_with_vocabulary_list("occupation", [str(x) for x in self.occupationList])
        occupation_column = tf.feature_column.embedding_column(occupation_column, 2)
        year_column = tf.feature_column.categorical_column_with_vocabulary_list("year", [str(x) for x in self.yearList])
        year_column = tf.feature_column.embedding_column(year_column, 3)
        # 多值特征
        genres_column = tf.feature_column.categorical_column_with_vocabulary_list("genres", self.genresList)
        genres_column = tf.feature_column.embedding_column(genres_column, 3)
        feature_columns = [gender_column, age_column, occupation_column, year_column, genres_column]
        # dense input
        input_layer = tf.feature_column.input_layer(features, feature_columns)
        dense_input = tf.identity(input_layer, name="dense_input")
        # experts
        experts_weight = tf.get_variable(name="experts_weight", dtype=tf.float32, shape=(dense_input.get_shape()[1], params.experts_units, params.experts_num),
                                         initializer=tf.contrib.layers.xavier_initializer())
        expert_bias = tf.get_variable(name="expert_bias", shape=(params.experts_units, params.experts_num),
                                      initializer=tf.contrib.layers.xavier_initializer())
        # gates
        gate1_weight = tf.get_variable(name="gate1_weight", dtype=tf.float32, shape=(dense_input.get_shape()[1], params.experts_num),
                                       initializer=tf.contrib.layers.xavier_initializer())
        gate1_bias = tf.get_variable(name="gate1_bias", dtype=tf.float32, shape=(params.experts_num),
                                     initializer=tf.contrib.layers.xavier_initializer())
        gate2_weight = tf.get_variable(name="gate2_weight", dtype=tf.float32, shape=(dense_input.get_shape()[1], params.experts_num),
                                       initializer=tf.contrib.layers.xavier_initializer())
        gate2_bias = tf.get_variable(name="gate2_bias", dtype=tf.float32, shape=(params.experts_num),
                                     initializer=tf.contrib.layers.xavier_initializer())
        # f_{i}(x) = activation(W_{i} * x + b), where activate is ReLU according to the paper
        experts_output = tf.tensordot(dense_input, experts_weight, axes=1, name="expert_outputs")
        use_expert_bias = True
        if use_expert_bias:
            experts_output = tf.add(experts_output, expert_bias)
        experts_output = tf.nn.relu(experts_output)

        # g^{k}(x) = activation(W_{gk}*x+b), where activation is softmax according to the paper
        gate1_output = tf.matmul(dense_input, gate1_weight)
        gate2_output = tf.matmul(dense_input, gate2_weight)
        use_gate_bias = True
        if use_gate_bias:
            gate1_output = tf.add(gate1_output, gate1_bias)
            gate2_output = tf.add(gate2_output, gate2_bias)
        gate1_output = tf.nn.softmax(gate1_output)
        gate2_output = tf.nn.softmax(gate2_output)

        # f^{k}(x) = sum_{i=1}^{n}(g^{k}(x)_{i} * f_{i}(x))
        label1_input = tf.multiply(experts_output, tf.expand_dims(gate1_output, axis=1))
        label1_input = tf.reduce_sum(label1_input, axis=2)
        label1_input = tf.reshape(label1_input, [-1, params.experts_units])
        label2_input = tf.multiply(experts_output, tf.expand_dims(gate2_output, axis=1))
        label2_input = tf.reduce_sum(label2_input, axis=2)
        label2_input = tf.reshape(label2_input, [-1, params.experts_units])
        len_layers = len(params.hidden_units)
        with tf.variable_scope("ctr_deep"):
            dense_ctr = tf.layers.dense(inputs=label1_input, units=params.hidden_units[0], activation=tf.nn.relu)
            for i in range(1, len_layers):
                dense_ctr = tf.layers.dense(intputs=dense_ctr, units=params.hidden_units[i], activation=tf.nn.relu)
            ctr_output = tf.layers.dense(inputs=dense_ctr, units=1)
        with tf.variable_scope("cvr_deep"):
            dense_cvr = tf.layers.dense(inputs=label2_input, units=params.hidden_units[0], activation=tf.nn.relu)
            for i in range(1, len_layers):
                dense_cvr = tf.layers.dense(inputs=dense_cvr, units=params.hidden_units[i], activation=tf.nn.relu)
            cvr_output = tf.layers.dense(inputs=dense_cvr, units=1)

        ctr_score = tf.identity(tf.nn.sigmoid(ctr_output), name="ctr_score")
        cvr_score = tf.identity(tf.nn.sigmoid(cvr_output), name="cvr_score")
        ctcvr_score = ctr_score*cvr_score
        ctcvr_score = tf.identity(ctcvr_score, name="ctcvr_score")

        score = tf.add(ctr_score * params.label1_weight, cvr_score*params.label2_weight)
        score = tf.identity(score, name="score")
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=score)
        else:
            ctr_labels = tf.identity(labels["label"], name="ctr_labels")
            ctcvr_labels = tf.identity(labels["label2"], name="ctcvr_labels")
            ctr_auc = tf.metrics.auc(labels=ctr_labels, predictions=ctr_score, name="ctr_auc")
            ctcvr_auc = tf.metrics.auc(labels=ctcvr_labels, predictions=ctcvr_score, name="ctcvr_auc")
            metrics = {"ctr_auc": ctr_auc, "ctcvr_auc": ctcvr_auc}
            ctr_loss = tf.reduce_mean(tf.losses.log_loss(labels=ctr_labels, predictions=ctr_score))
            ctcvr_loss = tf.reduce_mean(tf.losses.log_loss(labels=ctcvr_labels, predictions=ctcvr_score))
            loss = ctr_loss + ctcvr_loss
            if mode == tf.estimator.ModeKeys.TRAIN:
                optimizer = tf.train.AdamOptimizer(params.learning_rate)
                train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
            else :
                train_op = None
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=metrics, train_op=train_op)

    def train(self):
        model_estimator = self.model_estimator(self.params)
        # model.train(input_fn=self.train_input_fn, hooks=[tf.train.LoggingTensorHook(["input_layer", "ctr_score"], every_n_iter=500)])  ## input_layer
        train_spec = tf.estimator.TrainSpec(input_fn=lambda : self.train_input_fn(f="train"), hooks=[tf.train.LoggingTensorHook(["dense_input", "expert_outputs", "ctr_score"], every_n_iter=500)])
        eval_spec = tf.estimator.EvalSpec(input_fn=lambda : self.train_input_fn(f="test"), steps=None, start_delay_secs=1000, throttle_secs=1200)
        tf.estimator.train_and_evaluate(model_estimator, train_spec, eval_spec)

def main(_):
    params = {"embedding_size": 6, "feature_size": 0, "field_size": 0, "batch_size": 64, "learning_rate": 0.001,"epochs":200,
              "optimizer": "adam", "data_path": "../data/ml-1m/", "model_dir": "../data/model/mmoe/", "hidden_units":[8],
              "experts_units": 2, "experts_num":2, "label1_weight": 0.5, "label2_weight": 0.5}
    m = MMoEModel(configParam=ConfigParam(params))
    m.train()


if __name__ == '__main__':
    print(tf.__version__)
    tf.app.run()



