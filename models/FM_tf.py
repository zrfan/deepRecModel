# -*- coding: utf-8 -*
# tf1.14
from __future__ import division
from math import exp
from numpy import *
from random import normalvariate  # 正态分布
from sklearn import preprocessing
import numpy as np
from data_util import get1MTrainData
import tensorflow as tf
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 使用第0块GPU


# https://zhuanlan.zhihu.com/p/145436595
class FMModelParams(object):
    """ class for initializing weights"""

    def __init__(self, params):
        self.embedding_size, self.feature_size = params["embedding_size"], params["feature_size"]

    def initialize_weights(self):
        """ init fm weights
        returns
        weights: feature_embeddings: vi, vj second order params
                 weights_first_order: wi first order params
                 bias: b bias
        """
        weights = dict()
        weights_initializer = tf.glorot_normal_initializer()
        bias_initializer = tf.constant_initializer(0.0)
        weights["feature_embeddings"] = tf.get_variable(name="weights", dtype=tf.float32,
                                                        initializer=weights_initializer,
                                                        shape=[self.feature_size + 1, self.embedding_size])
        weights["weights_first_order"] = tf.get_variable(name="vector", dtype=tf.float32,
                                                         initializer=weights_initializer,
                                                         shape=[self.feature_size + 1, 1])
        weights["fm_bias"] = tf.get_variable(name="bias", dtype=tf.float32, initializer=bias_initializer, shape=[1])
        return weights


class FMModel(object):
    """ FM implementation for tensorflow"""

    def __init__(self, data_path, params):
        self.data_path, self.params = data_path, params

    def fm_model_fn(self, features, labels, mode):
        """ build tf model """
        embedding_size, feature_size, field_size = self.params["embedding_size"], self.params["feature_size"], \
                                                   self.params["field_size"]
        batch_size, learning_rate, optimizer_used = self.params["batch_size"], self.params["learning_rate"], \
                                                    self.params["optimizer"]
        feature_idx = features["feature_idx"]
        print("feature  :   ########################3\n", features)
        feature_idx = tf.reshape(feature_idx, shape=[batch_size, tf.shape(feature_idx)[1]])
        labels = tf.reshape(labels, shape=[batch_size, 1])
        feature_values = features["feature_values"]
        feature_values = tf.reshape(feature_values, shape=[batch_size, tf.shape(feature_idx)[1], 1])

        tf_model_params = FMModelParams(self.params)
        weights = tf_model_params.initialize_weights()
        embeddings = tf.nn.embedding_lookup(weights["feature_embeddings"], feature_idx)
        weights_first_order = tf.nn.embedding_lookup(weights["weights_first_order"], feature_idx)
        bias = weights["fm_bias"]
        # build function
        ## first order
        first_order = tf.multiply(feature_values, weights_first_order, name="first_order")
        first_order = tf.reduce_sum(first_order, 2)
        first_order = tf.reduce_sum(first_order, 1, keepdims=True)

        ## second order
        ### feature * embedding
        feature_emb = tf.multiply(feature_values, embeddings)
        ### square(sum(feature * embedding))
        feature_emb_sum = tf.reduce_sum(feature_emb, 1)
        feature_emb_sum_square = tf.square(feature_emb_sum)
        ### sum(square(feature * embedding))
        feature_emb_square = tf.square(feature_emb)
        feature_emb_square_sum = tf.reduce_sum(feature_emb_square, axis=1)

        second_order = feature_emb_sum_square - feature_emb_square_sum
        second_order = tf.reduce_sum(second_order, axis=1, keep_dims=True)

        ## final objective function
        logits = second_order + first_order + bias
        predicts = tf.sigmoid(logits)

        ## loss function
        sigmoid_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
        sigmoid_loss = tf.reduce_mean(sigmoid_loss, name="sigmoid_loss")
        loss = sigmoid_loss

        # train_op
        if optimizer_used == 'adagrad':
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
        elif optimizer_used == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        else:
            raise Exception("unknown optimizer", optimizer_used)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

        # metric
        eval_metric_ops = {"auc": tf.metrics.auc(labels, predicts)}
        predictions = {"prob": predicts}
        return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.TRAIN, predictions=predicts, loss=loss,
                                          eval_metric_ops=eval_metric_ops, train_op=train_op)

    def train_input_fn(self):
        userData, itemData, rating_info, user_cols, movie_cols = get1MTrainData(self.data_path)
        self.params["feature_size"] = len(user_cols) + len(movie_cols) - 2

        def gen():
            for idx, row in rating_info.iterrows():
                userId, itemId = int(row["userId"]), int(row["movieId"])
                userInfo, movieInfo = userData.loc[userId, :], itemData.loc[itemId, :]
                trainData = userInfo.tolist() + movieInfo.tolist()
                feature_index = list(filter(lambda x: x[0] == 1, zip(trainData, list(range(1, len(trainData) + 1)))))
                feature_index = list(map(lambda x: x[1], feature_index))
                feature_values = [1 for _ in range(len(feature_index))]
                y = float(row["ratings"]) / 5
                # print("feature_indx", feature_index, "features len", len(feature_index))
                feature_dict = {"feature_idx": feature_index, "feature_values": feature_values}
                yield (feature_dict, y)

        dataset = tf.data.Dataset.from_generator(gen,
                                                 ({"feature_idx": tf.int64, "feature_values": tf.float32}, tf.float32),
                                                 ({"feature_idx": tf.TensorShape([None]),
                                                   "feature_values": tf.TensorShape([None])},
                                                  tf.TensorShape([])))
        dataset = dataset.prefetch(self.params["batch_size"] * 10).padded_batch(self.params["batch_size"],
                                                                                padded_shapes=({"feature_idx": [None],
                                                                                                "feature_values": [
                                                                                                    None]}, []))
        return dataset

    def input_fn_test(self):
        userData, itemData, rating_info, user_cols, movie_cols = get1MTrainData(self.data_path)
        self.params["feature_size"] = len(user_cols) + len(movie_cols) - 2
        userIdx, userInfos = [], []
        for idx, row in userData.iterrows():
            userIdx.append(idx)
            userfeatures = list(filter(lambda x: x[1] == 1, zip(row, list(range(1, len(row) + 1)))))
            val = [str(x) for x in userfeatures]
            userInfos.append(','.join(val))
        print("user len=", len(userIdx))
        print(userInfos[:1])
        default_value = tf.constant("0", dtype=tf.string)
        usertable = tf.contrib.lookup.HashTable(
            tf.contrib.lookup.KeyValueTensorInitializer(userIdx, userInfos),
            default_value)
        itemIdx, itemInfos = [], []
        for idx, row in itemData.iterrows():
            itemIdx.append(idx)
            itemfeatures = list(filter(lambda x:x[1]==1, zip(row, list(range(len(user_cols), len(row)+len(user_cols))))))
            itemInfos.append(','.join([str(x) for x in itemfeatures]))
        itemtable = tf.contrib.lookup.HashTable(
            tf.contrib.lookup.KeyValueTensorInitializer(itemIdx, itemInfos),
            default_value)
        print("item=", itemInfos[:1])
        # data = []
        # for _, row in rating_info.iterrows():
        #     userId, itemId = row["userId"], row["movieId"]
        #     userInfo, movieInfo = userData.loc[userId, :], itemData.loc[itemId, :]
        #     trainData = userInfo.tolist() + movieInfo.tolist()
        #     feature_index = list(filter(lambda x: x[0] == 1, zip(trainData, list(range(1, len(trainData) + 1)))))
        #     feature_index = list(map(lambda x: str(x[1]), feature_index))
        #     # userIdx = list(filter(lambda x:x[1]==1, zip(userInfo, list(range(1, len(userInfo)+1)))))
        #     # itemIdx = list(filter(lambda x:x[1]==1, zip(movieInfo, list(range(0, len(movieInfo))))))
        #     y = float(row["ratings"]) / 5
        #     data.append([','.join(feature_index), y])
        # print("data len=", len(data))

        def decode(row):
            userId, itemId, label = tf.cast(row[0], dtype=tf.int32), tf.cast(row[1], dtype=tf.int32), row[2]
            print("######## user_id=\n", userId)
            userInfo = usertable.lookup(userId)
            print("######## user_info=\n", userInfo)
            itemInfo = itemtable.lookup(itemId)
            # all_features = tf.strings.to_number(tf.reshape(tf.sparse.to_dense(tf.string_split([userInfo, itemInfo], ","),
            #                                                               default_value="0"),
            #                                                 [-1]),
            #                                     out_type=tf.int32)
            # feature_index = all_features
            # print("#########     feature_index  ######=\n", feature_index)
            # feature_values = tf.ones_like(feature_index, dtype=tf.float32)
            # label = tf.div(tf.cast(label, tf.float32), 5)
            # # print("feature_indx", feature_index, "features len", len(feature_index))
            feature_dict = {"feature_idx": userInfo, "feature_values": itemInfo}
            return (feature_dict, label)

        dataset = tf.data.Dataset.from_tensor_slices(rating_info).map(decode, num_parallel_calls=2)
        # dataset = dataset.prefetch(self.params["batch_size"] * 10) \
        #     .padded_batch(self.params["batch_size"], padded_shapes=({"feature_idx": [None], "feature_values": [None]}, []))
        return dataset

    def train(self):

        session_config = tf.ConfigProto(log_device_placement=True, device_count={'GPU': 0})
        session_config.gpu_options.per_process_gpu_memory_fraction = 0.8
        config = tf.estimator.RunConfig(keep_checkpoint_max=2, log_step_count_steps=500, save_summary_steps=50,
                                        save_checkpoints_steps=50000).replace(session_config=session_config)

        fm_model = tf.estimator.Estimator(model_fn=self.fm_model_fn, model_dir="../data/model/", config=config)
        fm_model.train(input_fn=self.input_fn_test, hooks=[tf.train.LoggingTensorHook(["first_order", "sigmoid_loss"],
                                                                                      every_n_iter=500)])

    def test_dataset(self):
        dataset = self.input_fn_test().make_initializable_iterator()
        next_ele = dataset.get_next()
        batch = 1
        with tf.train.MonitoredTrainingSession() as sess:
            sess.run(dataset.initializer)
            while batch < 15:
                value = sess.run(next_ele)
                print("value=", value)
                batch += 1


def main(_):
    params = {"embedding_size": 8, "feature_size": 0, "field_size": 1, "batch_size": 64, "learning_rate": 0.001,
              "optimizer": "adam"}
    fm = FMModel(data_path="../data/ml-1m/", params=params)
    fm.test_dataset()
    # fm.train()


if __name__ == '__main__':
    print(tf.__version__)
    tf.app.run()
