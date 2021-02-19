# -*- coding: utf-8 -*
# tf1.14
import tensorflow as tf
import os
from data_util import get1MTrainData

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 使用第0块GPU

class FFMParams(object):
    def __init__(self, params):
        self.embedding_size, self.feature_size = params["embedding_size"], params["feature_size"]
        self.field_size = params["field_size"]
    def initialize_weights(self):
        weights = dict()
        weights_initializer = tf.glorot_normal_initializer()
        bias_initializer = tf.constant_initializer(0.0)
        weights["feature_embeddings"] = tf.get_variable(name="weights", dtype=tf.float32, initializer=weights_initializer,
                                                       shape=[self.feature_size+1, self.field_size, self.embedding_size])
        weights["weight_first_order"] = tf.get_variable(name="first_vector", dtype=tf.float32, initializer=weights_initializer,
                                                        shape=[self.feature_size+1, 1])
        weights["ffm_bias"] = tf.get_variable(name="bias", dtype=tf.float32, initializer=bias_initializer, shape=[1])
        return weights

# https://zhuanlan.zhihu.com/p/145928996
# https://github.com/wziji/deep_ctr
class FFMModel(object):
    """Field-aware of Factorization Machine implementation for tensorflow"""
    def __init__(self, data_path, params):
        self.data_path, self.params = data_path, params
    def ffm_model_fn(self, features, labels, mode):
        batch_size, learning_rate, optimizer_used = self.params["batch_size"], self.params["learning_rate"], self.params["optimizer"]
        feature_idx = features["feature_idx"]
        feature_idx = tf.reshape(feature_idx, shape=[batch_size, tf.shape(feature_idx)[1]])
        labels = tf.reshape(labels, shape=[batch_size, 1])
        feature_values = features["feature_values"]
        feature_values = tf.reshape(feature_values, shape=[batch_size, tf.shape(feature_values)[1], 1])
        feature_fields = features["feature_fields"]
        # feature_fields = tf.reshape(feature_fields, shape=[batch_size, tf.shape(feature_fields)[1], 1])

        ffm_params = FFMParams(self.params)
        all_weights = ffm_params.initialize_weights()
        bias = all_weights["bias"]
        all_embedding = all_weights["feature_embeddings"]
        weights_first_order = tf.nn.embedding_lookup(all_weights["weights_first_order"], feature_idx)


        ## first_order
        first_order = tf.multiply(weights_first_order, feature_values, name="first_order")
        first_order = tf.reduce_sum(first_order, 2)
        first_order = tf.reduce_sum(first_order, 1, keep_dims=True)

        ## second_order
        second_order = tf.constant(0, dtype=tf.float32)
        input_number = tf.shape(feature_idx).as_list()[1]
        for i in range(input_number):
            for j in range(i+1, input_number):
                idx_i, idx_j = feature_idx[:, i, :], feature_idx[:, j, :]
                field_i, field_j = feature_fields[:, i], feature_fields[:, j]
                emb_i, emb_j = all_embedding[idx_i, field_j, :], all_embedding[idx_j, field_i, :]
                val_i, val_j = feature_values[:, i, :], feature_values[:, j, :]

                field_emb_sum = tf.multiply(emb_i, emb_j)
                val_sum = tf.multiply(val_i, val_j)

                sum = tf.multiply(tf.reduce_sum(field_emb_sum, axis=1), val_sum)
                second_order += sum
        ## final objective function
        logits = second_order + first_order + bias
        predicts = tf.sigmoid(logits)

        ## loss function
        sigmoid_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
        sigmoid_loss = tf.reduce_mean(sigmoid_loss, name="sigmoid_loss")
        loss = sigmoid_loss

        # train_op
        if optimizer_used == "adagrad":
            optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
        elif optimizer_used == "adam":
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
        feature_dict = {"age": 0, "gender": 1, "occupation": 2, "genres": 3, "year": 4}
        self.params["feature_size"] = len(user_cols) + len(movie_cols) - 2
        self.params["field_size"] = len(feature_dict.keys())
        featureIdx, fieldIdx = [], []
        for col, idx in zip(user_cols+movie_cols, list(range(1, len(user_cols+movie_cols)+1))):
            featureIdx.append(idx)
            fieldIdx.append(feature_dict[col.split("_")[0]])
        fieldTable = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(featureIdx, fieldIdx),
                                                 tf.constant(0, dtype=tf.int32))
        def getTable(data, start_idx):
            idx, infos = [], []
            for idx, row in data.iterrows():
                idx.append(idx)
                features = list(filter(lambda x: x[0]==1, zip(row, list(range(start_idx, len(row)+1)))))
                val = [str(x[1]) for x in features]
                infos.append(','.join(val))
            print("data len=", len(idx))
            default_value = tf.constant("0", dtype=tf.string)
            table = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(idx, infos), default_value)
            return table
        userTable = getTable(userData, start_idx=1)
        itemTable = getTable(itemData, start_idx=len(user_cols)+1)
        def decode(row):
            userId, itemId, label = tf.cast(row[0], dtype=tf.int32), tf.cast(row[1], dtype=tf.int32), row[2]
            userInfo, itemInfo = userTable.lookup(userId), itemTable.lookup(itemId)
            all_features = tf.strings.to_number(tf.reshape(tf.sparse.to_dense(tf.strings.split([userInfo, itemInfo], ","),
                                                                              default_value="0"),
                                                           [-1]),
                                                out_type=tf.int32)
            feature_idx = all_features
            feature_fields = fieldTable.lookup(feature_idx)
            feature_values = tf.ones_like(feature_idx, dtype=tf.float32)
            label = tf.divide(tf.cast(label, tf.float32), 5)
            feature_dict = {"feature_idx": feature_idx, "feature_values": feature_values, "feature_fields": feature_fields}
            return (feature_dict, label)
        dataset = tf.data.Dataset.from_tensor_slices(rating_info).map(decode, num_parallel_calls=2)
        dataset = dataset.repeat(self.params["epochs"])
        dataset = dataset.prefetch(self.params["batch_size"]*10)\
            .padded_batch(self.params["batch_size"],
                          padded_shapes=({"feature_idx": [None], "feature_values": [None], "feature_fields": [None]}, []))
        return dataset
    def test_dataset(self):
        dataset = self.train_input_fn().make_initializable_iterator()
        next_ele = dataset.get_next()
        batch = 1
        with tf.train.MonitoredTrainingSession() as sess:
            sess.run(dataset.initializer)
            while batch < 15:
                value = sess.run(next_ele)
                print("value=", value)
                batch += 1

def main(_):
    params = {"embedding_size": 2, "feature_size": 0, "field_size": 0, "batch_size": 1, "learning_rate": 0.001,"epochs":200,
              "optimizer": "adam"}
    fm = FFMModel(data_path="../data/ml-1m/", params=params)
    fm.test_dataset()
    # fm.train()


if __name__ == '__main__':
    print(tf.__version__)
    tf.app.run()


