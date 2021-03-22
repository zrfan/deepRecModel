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
# Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate
##
class ESMMModel(BaseEstimatorModel):
    def __init__(self, configParam):
        self.params = configParam
        self.data_path = configParam.data_path
    def model_fn(self, features, labels, mode, params):
        # sparse_feature_size, multi_feature_size, embedding_size = self.params["sparse_feature_size"], self.params["multi_feature_size"], self.params["embedding_size"]
        # batch_size, learning_rate, optimizer_used = self.params["batch_size"], self.params["learning_rate"], self.params["optimizer"]
        feature_dict = {"gender": 0, "age": 0, "occupation": 0, "genres": 1, "year": 1}
        # ageList = [1, 18, 25, 35, 45, 50, 56]
        # occupationList = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
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
        #
        input_layer = tf.feature_column.input_layer(features, feature_columns)

        # # dense input dense
        # dense_input = tf.concat([sparse_emb, multi_mean_emb], axis=1, name="dense_vector")
        dense_input = tf.identity(input_layer, name="inputlayer")
        len_layers = len(params.hidden_units)
        with tf.variable_scope("ctr_deep"):
            dense_ctr = tf.layers.dense(dense_input, units=params.hidden_units[0], activation=tf.nn.relu)
            for i in range(1, len_layers):
                dense_ctr = tf.layers.dense(dense_ctr, units=params.hidden_units[i], activation=tf.nn.relu)
            ctr_out = tf.layers.dense(dense_ctr, units=1)
        with tf.variable_scope("cvr_deep"):
            dense_cvr = tf.layers.dense(dense_input, units=params.hidden_units[0], activation=tf.nn.relu)
            for i in range(1, len_layers):
                dense_cvr = tf.layers.dense(dense_cvr, units=params.hidden_units[i], activation=tf.nn.relu)
            cvr_output = tf.layers.dense(dense_cvr, units=1)
        ctr_score = tf.identity(tf.nn.sigmoid(ctr_out), name="ctr_score")
        cvr_score = tf.identity(tf.nn.sigmoid(cvr_output), name="cvr_score")
        ctcvr_score = tf.identity(ctr_score*cvr_score, name="ctcvr_score")
        ctr_pow, cvr_pow = 0.5, 1
        score = tf.multiply(tf.pow(ctr_score, ctr_pow), tf.pow(cvr_score, cvr_pow))
        predicts = tf.identity(score, name="score")
        ctr_labels, ctcvr_labels = tf.identity(labels["label"], name="ctr_labels"), tf.identity(labels["label2"], name="ctcvr_labels")
        print("***********ctr_labels=", ctr_labels)
        print("***********ctr_scores=", ctr_score)
        ctr_auc = tf.metrics.auc(labels=ctr_labels, predictions=ctr_score, name="ctr_auc")
        ctcvr_auc = tf.metrics.auc(labels=ctcvr_labels, predictions=ctcvr_score, name="ctcvr_auc")
        metrics = {"ctr_auc": ctr_auc, "ctcvr_auc": ctcvr_auc}
        ctr_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=ctr_labels, logits=ctr_out))
        ctcvr_loss = tf.reduce_mean(tf.losses.log_loss(labels=ctcvr_labels, predictions=ctcvr_score))
        loss = ctr_loss + ctcvr_loss
        tf.summary.scalar("loss", loss)
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        # if mode == tf.estimator.ModeKeys.PREDICT:
        #     return tf.estimator.EstimatorSpec(mode=mode, predictions=score)
        # else:
        #     if mode == tf.estimator.ModeKeys.TRAIN:
        #     else:
        #         train_op = None
        #     return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=metrics, train_op=train_op)
        # metric
        eval_metric_ops = {"auc": tf.metrics.auc(labels["label"], predicts)}
        predictions = {"prob": predicts}

        export_outputs = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(predictions)}

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)
        elif mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=loss, eval_metric_ops=eval_metric_ops)
        elif mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=loss, eval_metric_ops=eval_metric_ops, train_op=train_op)

    def train(self):
        # summary_hook = tf.train.SummarySaveHook(100, output_dir=self.params.model_dir+"/../summary/", summary_op=tf.summary.merge_all())
        profile_hook = tf.estimator.ProfilerHook(save_steps=5000, output_dir=self.params.model_dir)
        # with tf.contrib.tfprof.ProfileContext(self.params.model_dir+"/../profile/") as pctx:
        model_estimator = self.model_estimator(self.params)
        # model.train(input_fn=self.train_input_fn, hooks=[tf.train.LoggingTensorHook(["inputlayer", "ctr_score"], every_n_iter=500)])
        train_spec = tf.estimator.TrainSpec(input_fn=lambda : self.train_origin_input_fn(f="train"),  hooks=[profile_hook])
        eval_spec = tf.estimator.EvalSpec(input_fn=lambda : self.train_origin_input_fn(f="test"), steps=None, start_delay_secs=1000, throttle_secs=1200)
        tf.estimator.train_and_evaluate(model_estimator, train_spec, eval_spec)
    def test_run_dataset(self):
        # self.get_dataset()
        dataset = self.train_origin_input_fn(f="train").make_initializable_iterator()
        features, labels = dataset.get_next()
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
        genres_column = tf.feature_column.categorical_column_with_vocabulary_list("genres", self.genresList, default_value=-1)
        genres_column = tf.feature_column.embedding_column(genres_column, 3)
        # gender_column, age_column, occupation_column, year_column,
        feature_columns = [gender_column, age_column, occupation_column, year_column, genres_column]
        #
        input_layer = tf.feature_column.input_layer(features, feature_columns)
        print("*************input_layer=", input_layer)
        batch = 1
        with tf.train.MonitoredTrainingSession() as sess:
            sess.run(dataset.initializer)
            while batch <= 2000:
                value = sess.run([features, input_layer])
                print("value=", value)
                batch += 1

def main(_):
    params = {"embedding_size": 6, "feature_size": 0, "field_size": 0, "batch_size": 128, "learning_rate": 0.001,"epochs":200,
              "optimizer": "adam", "data_path": "../data/ml-1m/", "model_dir": "../data/model/essm/", "hidden_units":[8]}
    m = ESMMModel(configParam=ConfigParam(params))
    # m.test_run_dataset()
    m.train()


if __name__ == '__main__':
    print(tf.__version__)
    tf.app.run()











