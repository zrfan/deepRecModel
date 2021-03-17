# -*- coding: utf-8 -*
# tf1.14
import tensorflow as tf
import abc
from models.model_util import registerAllFeatureHashTable, registerAllFeatureIdxHashTable
from models.data_util import get1MTrainDataOriginFeatures, get1MTrainData
import random

class BaseEstimatorModel(object):
    def __init__(self):
        pass
    @abc.abstractmethod
    def model_fn(self):
        raise NotImplementedError
    def model_estimator(self, params):
        tf.reset_default_graph()
        session_config = tf.ConfigProto(log_device_placement=False, device_count={'GPU': 0})
        session_config.gpu_options.per_process_gpu_memory_fraction = 0.8
        config = tf.estimator.RunConfig(keep_checkpoint_max=2, log_step_count_steps=500, save_summary_steps=50,
                                        save_checkpoints_steps=50000).replace(session_config=session_config)
        model_estimator = tf.estimator.Estimator(model_fn=self.model_fn, model_dir=params.model_dir, params=params, config=config)
        return model_estimator

    def get_origin_dataset(self):
        self.userData, self.itemData, self.train_rating_info, self.test_rating_info, self.user_cols, self.movie_cols,ageList, occupationList, genresList, yearList \
            = get1MTrainDataOriginFeatures(self.params.data_path)
        self.ageList, self.occupationList, self.genresList, self.yearList = ageList, occupationList, genresList, yearList

        print(self.itemData.head(10))
        self.feature_dict = {"gender": ["F", "M"], "age": self.ageList, "occupation": self.occupationList, "genres": self.genresList, "year": self.yearList}
        self.params.feature_size = len(self.user_cols) + len(self.movie_cols)
        self.all_feature_hashtable, self.ulen = registerAllFeatureHashTable(self.userData, self.itemData)

    def train_origin_input_fn(self, f="train"):
        # if not hasattr(self, 'all_feature_hashtable'):
        self.get_origin_dataset()
        def decode(row):
            userId, itemId, label = tf.cast(row[0], dtype=tf.int32), tf.cast(row[1], dtype=tf.int32), tf.cast(row[2], dtype=tf.float32)
            userInfo, itemInfo = self.all_feature_hashtable.lookup(userId), self.all_feature_hashtable.lookup(tf.add(itemId, tf.constant(self.ulen, dtype=tf.int32)))
            user_features = tf.reshape(tf.sparse.to_dense(tf.strings.split([userInfo], ","), default_value="0"), shape=[-1, 1])
            item_features = tf.reshape(tf.sparse.to_dense(tf.strings.split([itemInfo], ","), default_value="0"), shape=[-1, 1])
            genres = tf.reshape(tf.sparse.to_dense(tf.strings.split([item_features[0]], "|"), default_value="0"), shape=[1, -1])
            #
            feature_dict = {"item_features": item_features, "genres": genres, "year": item_features[1],
                            "userId": userId, "itemId": itemId,
                            "gender": user_features[0], "age": user_features[1], "occupation": user_features[2],
                            "user_features": user_features}
            label = tf.divide(label, 5)
            label2 = tf.constant(random.random(), dtype=tf.float32)
            return feature_dict, {"label": [[label]], "label2": [[label2]]}
        # if not hasattr(self, "train_dataset"):
        #     self.train_dataset = tf.data.Dataset.from_tensor_slices(self.train_rating_info).map(decode, num_parallel_calls=2).repeat(self.params.epochs)
        #     self.test_dataset = tf.data.Dataset.from_tensor_slices(self.test_rating_info).map(decode, num_parallel_calls=2)
        if f=='train':
            dataset = tf.data.Dataset.from_tensor_slices(self.train_rating_info).map(decode, num_parallel_calls=2).repeat(self.params.epochs)
        else:
            dataset = tf.data.Dataset.from_tensor_slices(self.test_rating_info).map(decode, num_parallel_calls=2)
        return dataset
    def train_onehot_input_fn(self, f="train"):
        if not hasattr(self, "onehot_feature_hashtable"):
            userData, itemData, self.onehot_train_rating_info, self.onehot_test_rating_info, user_cols, movie_cols = get1MTrainData(self.params.data_path)
            self.params.feature_size = len(user_cols) + len(movie_cols)
            print("user_cols=", user_cols, " movie_cols=", movie_cols)
            self.onehot_feature_hashtable, self.user_feature_len = registerAllFeatureIdxHashTable(userData, itemData)

        def decode(row):
            userId, itemId, label = tf.cast(row[0], dtype=tf.int32), tf.cast(row[1], dtype=tf.int32), row[2]
            userInfo = self.onehot_feature_hashtable.lookup(userId)
            itemInfo = self.onehot_feature_hashtable.lookup(tf.add(itemId, tf.constant(self.user_feature_len, dtype=tf.int32)))
            all_features = tf.strings.to_number(tf.reshape(tf.sparse.to_dense(tf.string_split([userInfo, itemInfo], ","),
                                                                              default_value="0"),
                                                           [-1]),
                                                out_type=tf.int32)
            feature_index = all_features
            feature_values = tf.ones_like(feature_index, dtype=tf.float32)
            label = tf.div(tf.cast(label, tf.float32), 5)
            feature_dict = {"feature_idx": feature_index, "feature_values": feature_values, "userInfo": userInfo,
                            "itemInfo": itemInfo, "itemId": itemId, "userId": userId}
            return (feature_dict, label)
        shape = {"feature_idx": [None], "feature_values": [None],
                 "userInfo": [], "itemInfo": [], "userId": [], "itemId": []}
        if f == 'train':
            dataset = tf.data.Dataset.from_tensor_slices(self.onehot_train_rating_info).map(decode, num_parallel_calls=2)\
                .repeat(self.params.epochs)\
                .padded_batch(self.params.batch_size, padded_shapes=(shape, []))
        else:
            dataset = tf.data.Dataset.from_tensor_slices(self.onehot_test_rating_info).map(decode, num_parallel_calls=2) \
                .padded_batch(self.params.batch_size, padded_shapes=(shape, []))

        return dataset