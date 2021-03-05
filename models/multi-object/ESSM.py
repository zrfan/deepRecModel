# -*- coding: utf-8 -*
# tf1.14
import tensorflow as tf
import os
import sys
sys.path.append("../")
from models.data_util import get1MTrainDataOriginFeatures
from models.base_estimator_model import BaseEstimatorModel
from models.model_util import registerAllFeatureHashTable

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 使用第0块GPU

class EssmParams(object):
    def __init__(self, params):
        self.params = params
        self.seed, self.sparse_feature_size, self.embedding_size = params["seed"], params["sparse_feature_size"], params["embedding_size"]
    def initialize_weights(self):
        weights_initializer = tf.glorot_normal_initializer(seed=self.seed)
        bias_initializer = tf.constant(0,  dtype=tf.float32)
        bias = tf.get_variable(name='bias', dtype=tf.float32, initializer=bias_initializer, shape=[1])
        sparse_embeddings = tf.get_variable(name="sparse_embeddings", dtype=tf.float32, initializer=weights_initializer, shape=[self.sparse_feature_size, self.embedding_size])
        multi_embeddings = tf.get_variable(name="multi_embeddings", dtype=tf.float32, initializer=weights_initializer, shape=[self.multi_feature_size, self.embedding_size])
        return {"sparse_embddings": sparse_embeddings, "multi_embeddings": multi_embeddings}

class ESSMModel(BaseEstimatorModel):
    def __init__(self, params):
        self.params = params
        self.data_path = params["data_path"]
    def model_fn(self, features, labels, mode, params):
        sparse_feature_size, multi_feature_size, embedding_size = self.params["sparse_feature_size"], self.params["multi_feature_size"], self.params["embedding_size"]
        batch_size, learning_rate, optimizer_used = self.params["batch_size"], self.params["learning_rate"], self.params["optimizer"]
        feature_dict = {"gender": 0, "age": 0, "occupation": 0, "genres": 1, "year": 1}
        # ageList = [1, 18, 25, 35, 45, 50, 56]
        # occupationList = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        gender_column = tf.feature_column.categorical_column_with_identity("gender", num_buckets=2, default_value=0)
        gender_column = tf.feature_column.indicator_column(gender_column)
        age_column = tf.feature_column.categorical_column_with_vocabulary_list("age", self.ageList)
        age_column = tf.feature_column.indicator_column(age_column)
        occupation_column = tf.feature_column.categorical_column_with_identity("occupation", num_buckets=21, default_value=0)
        occupation_column = tf.feature_column.indicator_column(occupation_column)
        year_column = tf.feature_column.categorical_column_with_vocabulary_list("year", self.yearList)
        year_column = tf.feature_column.indicator_column(year_column)
        feature_columns = [gender_column, age_column, occupation_column, year_column]

        input_layer = tf.feature_column.input_layer(features, feature_columns)

        # sparse_feature_idx, multi_feature_idx = features["sparse_feature_idx"], features["multi_feature_idx"]
        # all_weights = EssmParams(self.params).initialize_weights()
        # sparse_embeddings, multi_embeddings = all_weights["sparse_embeddings"], all_weights["multi_embeddings"]
        # # one-hot category -> embedding
        # sparse_emb = tf.nn.embedding_lookup(sparse_embeddings, sparse_feature_idx)
        # sparse_emb = tf.reshape(sparse_emb, shape=[-1, sparse_feature_size*embedding_size])
        # # multi-hot category -> embedding
        # multi_emb = tf.nn.embedding_lookup(multi_embeddings, multi_feature_idx)  # [batch, len, emb_size]
        # sum_axis2 = tf.reduce_sum(multi_emb, axis=2)
        # nonzero = tf.count_nonzero(sum_axis2, keepdims=True, dtype=float)
        # multi_mean_emb = tf.div_no_nan(tf.reduce_sum(multi_emb, axis=1), nonzero)
        #
        # # dense input dense
        # dense_input = tf.concat([sparse_emb, multi_mean_emb], axis=1, name="dense_vector")
        input_layer = tf.identity(input_layer, name="inputlayer")
        dense_input = tf.concat(input_layer, axis=1, name="dense_vecotr")
        len_layers = len(params["hidden_units"])
        with tf.variable_scope("ctr_deep"):
            dense_ctr = tf.layers.dense(inputs=dense_input, units=params["hidden_units"][0], activation=tf.nn.relu)
            for i in range(1, len_layers):
                dense_ctr = tf.layers.dense(intputs=dense_ctr, units=params["hidden_units"][i], activation=tf.nn.relu)
            ctr_out = tf.layers.dense(inputs=dense_ctr, units=1)
        with tf.variable_scope("cvr_deep"):
            dense_cvr = tf.layers.dense(inputs=dense_input, units=params["hidden_units"][0], activation=tf.nn.relu)
            for i in range(1, len_layers):
                dense_cvr = tf.layers.dense(inputs=dense_cvr, units=params["hidden_units"][i], activation=tf.nn.relu)
            cvr_output = tf.layers.dense(inputs=dense_cvr, units=1)
        ctr_score = tf.identity(tf.nn.sigmoid(ctr_out), name="ctr_score")
        cvr_score = tf.identity(tf.nn.sigmoid(cvr_output), name="cvr_score")
        ctcvr_score = tf.identity(ctr_score*cvr_score, name="ctcvr_score")
        ctr_pow, cvr_pow = 0.5, 1
        score = tf.multiply(tf.pow(ctr_score, ctr_pow), tf.pow(cvr_score, cvr_pow))
        score = tf.identity(score, name="score")
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=score)
        else :
            ctr_labels, ctcvr_labels = tf.identity(labels["label"], name="ctr_labels"), tf.identity(labels["label2"], name="ctcvr_labels")
            ctr_auc = tf.metrics.auc(labels=ctr_labels, predictions=ctr_score, name="ctr_auc")
            ctcvr_auc = tf.metrics.auc(labels=ctcvr_labels, predictions=ctcvr_score, name="ctcvr_auc")
            metrics = {"ctr_auc": ctr_auc, "ctcvr_auc": ctcvr_auc}
            ctr_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=ctr_labels, logits=ctr_out))
            ctcvr_loss = tf.reduce_mean(tf.losses.log_loss(labels=ctcvr_labels, predictions=ctcvr_score))
            loss = ctr_loss + ctcvr_loss
            if mode == tf.estimator.ModeKeys.TRAIN:
                optimizer = tf.train.AdamOptimizer(params["learning_rate"])
                train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
            else:
                train_op = None
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=metrics, train_op=train_op)
    def get_dataset(self, params):
        userData, itemData, train_rating_info, test_rating_info, user_cols, movie_cols,ageList, occupationList, genresList, yearList\
            = get1MTrainDataOriginFeatures(self.data_path)
        self.ageList, self.occupationList, self.genresList, self.yearList = ageList, occupationList, genresList, yearList

        print(itemData.head(10))
        feature_dict = {"gender": 0, "age": 0, "occupation": 0, "genres": 1, "year": 1}
        self.params["feature_size"] = len(user_cols) + len(movie_cols)
        all_feature_hashtable, ulen = registerAllFeatureHashTable(userData, itemData)
        def decode(row):
            userId, itemId, label = tf.cast(row[0], dtype=tf.int32), tf.cast(row[1], dtype=tf.int32), tf.cast(row[2], dtype=tf.float32)
            userInfo, itemInfo = all_feature_hashtable.lookup(userId), all_feature_hashtable.lookup(tf.add(itemId, tf.constant(ulen, dtype=tf.int32)))
            user_features = tf.reshape(tf.sparse.to_dense(tf.strings.split([userInfo], ","), default_value="0"), shape=[-1])
            item_features = tf.reshape(tf.sparse.to_dense(tf.strings.split([itemInfo], ","), default_value="0"), shape=[-1])
            genres = tf.reshape(tf.sparse.to_dense(tf.strings.split([item_features[0]], "|"), default_value="0"), shape=[-1])
            feature_dict = {"item_features": item_features, "gender": user_features[0], "age": user_features[1],
                            "occupation": user_features[2], "genres": genres, "year": item_features[1],
                            "userId": userId, "itemId": itemId}
            label = tf.divide(label, 5)
            return feature_dict, label

        train_dataset = tf.data.Dataset.from_tensor_slices(train_rating_info).map(decode, num_parallel_calls=2).repeat(params["epochs"])
        test_dataset = tf.data.Dataset.from_tensor_slices(train_rating_info).map(decode, num_parallel_calls=2).repeat(params["epochs"])
        self.train_dataset, self.test_dataset = train_dataset, test_dataset
    def test_run_dataset(self, params):
        self.get_dataset(params)
        dataset = self.train_dataset.make_initializable_iterator()
        next_ele = dataset.get_next()
        batch = 1
        with tf.train.MonitoredTrainingSession() as sess:
            sess.run(dataset.initializer)
            while batch <= 20:
                value = sess.run(next_ele)
                print("value=", value)
                batch += 1
    def train_input_fn(self):
        if not hasattr(self, 'train_dataset'):
            self.get_dataset(self.params)
        return self.train_dataset
    def test_input_fn(self):
        if not hasattr(self, 'test_dataset'):
            self.get_dataset(self.params)
        return self.test_dataset
    def train(self):
        model = self.model_estimator(self.params)
        model.train(input_fn=self.train_input_fn, hooks=[tf.train.LoggingTensorHook(["inputlayer","dense_vecotr", "ctr_score"], every_n_iter=500)])

def main(_):
    params = {"embedding_size": 6, "feature_size": 0, "field_size": 0, "batch_size": 64, "learning_rate": 0.001,"epochs":200,
              "optimizer": "adam", "data_path": "../data/ml-1m/", "model_dir": "../data/model/essm/"}
    m = ESSMModel(params=params)
    # m.test_run_dataset(params)
    m.train()


if __name__ == '__main__':
    print(tf.__version__)
    tf.app.run()











