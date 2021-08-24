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


## https://github.com/ChenglongChen/tensorflow-DeepFM/blob/master/DeepFM.py
## https://zhuanlan.zhihu.com/p/48057256

class DeepFMModel(BaseEstimatorModel):
    def __init__(self, configParam):
        self.params = configParam
    def model_fn(self, features, labels, mode, params):
        embedding_size = self.params.embedding_size
        learning_rate, optimizer_used = self.params.learning_rate, self.params.optimizer
        embed_columns, bias_columns = [], []
        for key in self.feature_dict.keys():
            column = tf.feature_column.categorical_column_with_vocabulary_list(key, self.feature_dict[key])
            embed_columns.append(tf.feature_column.embedding_column(column, embedding_size))
            bias_columns.append(tf.feature_column.embedding_column(column, 1))
        embed_input_layer = tf.feature_column.input_layer(features, embed_columns)
        bias_input_layer = tf.feature_column.input_layer(features, bias_columns)


        first_order = tf.identity(bias_input_layer, name="first_order")
        embeddings = tf.identity(tf.reshape(embed_input_layer, shape=[-1, len(self.feature_dict.keys()), embedding_size]),
                                 name="fm_2_input")
        feature_emb_sum = tf.reduce_sum(embeddings, 1)
        feature_emb_sum_square = tf.square(feature_emb_sum)
        feature_emb_square = tf.square(embeddings)
        feature_emb_square_sum = tf.reduce_sum(feature_emb_square, 1)
        second_order = feature_emb_sum_square - feature_emb_square_sum
        second_order = tf.identity(second_order, name="second_order")

        deep = tf.identity(embed_input_layer, name="dense_input")
        weights_initializer = tf.glorot_normal_initializer()
        for i in range(0, len(self.params.deep_layers)):
            deep = tf.layers.dense(deep, units=self.params.deep_layers[i], activation=None,
                                   kernel_initializer=weights_initializer,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(0.00001))
            if self.params.batch_norm:
                deep = self.batch_norm_layer(deep, tf.constant(mode==tf.estimator.ModeKeys.TRAIN, dtype=tf.bool), scope_bn="bn_%d"%i)
            deep = tf.nn.relu(deep)
            deep = tf.nn.dropout(deep, self.params.dropout_keep_deep[i+1])
        print("########### first_order=", first_order, " second_order=", second_order, " deep=", deep)
        if self.params.use_fm and self.params.use_deep:
            concat_input = tf.concat([first_order, second_order, deep], axis=1)
        elif self.params.use_fm:
            concat_input = tf.concat([first_order, second_order], axis=1)
        elif self.params.use_deep:
            concat_input = deep
        print("############ concat_input=", concat_input)
        predicts = tf.layers.dense(concat_input, 1, activation=None,
                                   kernel_initializer=weights_initializer,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(0.00001))
        logits = tf.nn.sigmoid(predicts)
        #### loss
        if self.params.loss_type == "logloss":
            predicts = tf.nn.sigmoid(predicts)
            loss = tf.reduce_mean(tf.losses.log_loss(labels["label"], predicts))
        elif self.params.loss_type == "mse":
            loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels["label"], logits=predicts))
        print("######### loss=", loss)
        ### l2 regularization on weights
        # if self.params.l2_reg > 0:
        #     loss = tf.add_n([loss] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        print("######### loss=", loss)
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
        eval_metric_ops = {"auc": tf.metrics.auc(labels["label"], logits)}
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


    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = tf.contrib.layers.batch_norm(x, decay=self.params.batch_norm_decay, center=True, scale=True,
                                                       updates_collections=None, is_training=True, reuse=None,
                                                       trainable=True, scope=scope_bn)
        bn_inference = tf.contrib.layers.batch_norm(x, decay=self.params.batch_norm_decay, center=True, scale=True,
                                                                  updates_collections=None, is_training=False, reuse=True,
                                                                  trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda : bn_train, lambda : bn_inference)
        return z
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
        input_size = self.params.feature_size * self.params.embedding_size
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
            input_size = self.params.feature_size + self.params.embedding_size + self.params.deep_layers[-1]
        elif self.params.use_fm:
            input_size = self.params.feature_size + self.params.embedding_size
        elif self.params.use_deep:
            input_size = self.params.deep_layers[-1]
        weights["concat_projection"] = tf.get_variable(name="concat_projection", dtype=tf.float32,
                                                       initializer=weights_initializer,
                                                       shape=[input_size, 1])
        weights["concat_bias"] = tf.get_variable(name="concat_bias", dtype=tf.float32,
                                                 initializer=bias_initializer, shape=[1])
        return weights
    def new_model_fn(self, features, labels, mode, params):
        batch_size, learning_rate, optimizer_used = self.params.batch_size, self.params.learning_rate, self.params.optimizer
        feature_idx, feature_values = features["feature_idx"], features["feature_values"]
        print("feature_idx=", feature_idx)
        feature_idx = tf.reshape(feature_idx, shape=[-1, tf.shape(feature_idx)[1]])
        feature_values = tf.reshape(feature_values, shape=[-1, tf.shape(feature_idx)[1], 1])
        labels = tf.reshape(labels, shape=[-1, 1])
        weights = self._initialize_weights()
        embeddings = tf.nn.embedding_lookup(weights["feature_embeddings"], feature_idx)

        #### ---------------- first order term -------------
        feature_bias = tf.nn.embedding_lookup(weights["feature_bias"], feature_idx)
        first_order = tf.multiply(feature_bias, feature_values, name="first_order")

        print("############### first order=", first_order, " feature_idx=", feature_idx, " embeddings=", embeddings)
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
        deep = tf.reshape(embeddings, shape=[-1, self.params.feature_size * self.params.embedding_size])
        deep = tf.nn.dropout(deep, self.params.dropout_keep_deep[0])
        for i in range(0, len(self.params.deep_layers)):
            deep = tf.matmul(deep, weights["layer_%d"%i]) + weights["bias_%d"%i]
            if self.params.batch_norm:
                deep = self.batch_norm_layer(deep, tf.constant(mode==tf.estimator.ModeKeys.TRAIN, dtype=tf.bool), scope_bn="bn_%d"%i)
            deep = tf.nn.relu(deep)
            deep = tf.nn.dropout(deep, self.params.dropout_keep_deep[i+1])
        print("########### first_order=", first_order, " second_order=", second_order, " deep=", deep)
        ##### -------------- DeepFM -------------
        if self.params.use_fm and self.params.use_deep:
            concat_input = tf.concat([first_order, second_order, deep], axis=1)
        elif self.params.use_fm:
            concat_input = tf.concat([first_order, second_order], axis=1)
        elif self.params.use_deep:
            concat_input = deep
        print("############ concat_input=", concat_input, " labels=", labels["label"])
        predicts = tf.matmul(concat_input, weights["concat_projection"]) + weights["concat_bias"]
        #### loss
        if self.params.loss_type == "logloss":
            predicts = tf.nn.sigmoid(predicts)
            loss = tf.losses.log_loss(labels["label"], predicts)
        elif self.params.loss_type == "mse":
            loss = tf.nn.l2_loss(labels["label"] - predicts)
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


    def train(self):
        model_estimator = self.model_estimator(self.params)
        # model.train(input_fn=self.train_input_fn, hooks=[tf.train.LoggingTensorHook(["inputlayer", "ctr_score"], every_n_iter=500)])
        train_spec = tf.estimator.TrainSpec(input_fn=lambda: self.train_origin_input_fn(f="train"),
                                            hooks=[tf.train.LoggingTensorHook(["first_order", "second_order", "dense_input"], every_n_iter=500)])





        eval_spec = tf.estimator.EvalSpec(input_fn=lambda: self.train_origin_input_fn(f="test"), steps=None, start_delay_secs=1000, throttle_secs=1200)
        tf.estimator.train_and_evaluate(model_estimator, train_spec, eval_spec)
    def test_run_dataset(self):
        dataset = self.train_onehot_input_fn(f="train").make_initializable_iterator()
        features, labels = dataset.get_next()
        batch = 1
        with tf.train.MonitoredTrainingSession() as sess:
            sess.run(dataset.initializer)
            while batch <= 20:
                value = sess.run([features, labels])
                print("value=", value)
                batch += 1

def main(_):
    params = {"embedding_size": 4, "feature_size": 0, "field_size": 5, "batch_size": 64, "learning_rate": 0.001,"epochs":200, "l2_reg": 0.001,
              "optimizer": "adam", "data_path": "../data/ml-1m/", "model_dir": "../data/model/deepFm/",
              "dropout_keep_deep": [0.8, 1.0, 1.0, 1.0], "deep_layers": [64, 8], "dropout_keep_fm": [1.0, 1.0],
              "loss_type": "mse",
               "use_fm": True, "use_deep": True, "batch_norm": False, "batch_norm_decay": 0.995}
    m = DeepFMModel(configParam=ConfigParam(params))
    # m.test_run_dataset()
    m.train()


if __name__ == '__main__':
    print(tf.__version__)
    tf.set_random_seed(2019)
    tf.reset_default_graph()
    tf.app.run()


