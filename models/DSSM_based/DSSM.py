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

##
# DSSM
##
def get_cosine_similarity(user, ad):
    user_norm = tf.sqrt(tf.reduce_sum(tf.multiply(user, user), axis=1))
    ad_norm = tf.sqrt(tf.reduce_sum(tf.multiply(ad, ad), axis=1))


class DSSMModel(BaseEstimatorModel):
    def __init__(self, configParam):
        self.params = configParam
        self.data_path = configParam.data_path
    def model_fn(self, features, labels, mode, params):
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
        user_feature_columns = [gender_column, age_column, occupation_column, year_column]
        ad_feature_columns = [genres_column]
        #
        user_input_layer = tf.feature_column.input_layer(features, user_feature_columns)
        ad_input_layer = tf.feature_column.input_layer(features, ad_feature_columns)
        user_dense_input = tf.identity(user_input_layer, name="user_inputlayer")
        ad_dense_input = tf.identity(ad_input_layer, name="ad_inputlayer")

        len_layers = len(params.hidden_units)
        with tf.variable_scope("user_deep"):
            user_dense = tf.layers.dense(user_dense_input, units=params.hidden_units[0], activation=tf.nn.relu)
            for i in range(1, len_layers):
                user_dense = tf.layers.dense(user_dense, units=params.hidden_units[i], activation=tf.nn.relu)
            user_out = tf.layers.dense(user_dense, units=1)
        with tf.variable_scope("ad_deep"):
            ad_dense = tf.layers.dense(ad_dense_input, units=params.hidden_units[0], activation=tf.nn.relu)
            for i in range(1, len_layers):
                ad_dense = tf.layers.dense(ad_dense, units=params.hidden_units[i], activation=tf.nn.relu)
            ad_out = tf.layers.dense(ad_dense, units=1)

        # cosine_similarity


