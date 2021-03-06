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

class PLE(BaseEstimatorModel):
    def __init__(self, configParam):
        self.params = configParam
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
        feature_columns = [gender_column, age_column, occupation_column, year_column]
        # dense input
        input_layer = tf.feature_column.input_layer(features, feature_columns)
        dense_input = tf.identity(input_layer, name="dense_input")

