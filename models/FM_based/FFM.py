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
        weights["weights_first_order"] = tf.get_variable(name="first_vector", dtype=tf.float32, initializer=weights_initializer,
                                                        shape=[self.feature_size+1, 1])
        weights["ffm_bias"] = tf.get_variable(name="bias", dtype=tf.float32, initializer=bias_initializer, shape=[1])
        return weights

class FFMModel(object):
    """Field-aware of Factorization Machine implementation for tensorflow"""
    def __init__(self, params):
        self.params = params
        self.data_path = params["data_path"]
    def ffm_model_fn(self, features, labels, mode, params):
        feature_size = self.params["feature_size"]
        batch_size, learning_rate, optimizer_used = self.params["batch_size"], self.params["learning_rate"], self.params["optimizer"]
        origin_feature = features["origin_feature"]
        feature_idx = features["feature_idx"]
        feature_idx = tf.reshape(feature_idx, shape=[batch_size, tf.shape(feature_idx)[1]], name="feature_idx")
        labels = tf.reshape(labels, shape=[batch_size, 1])
        feature_values = features["feature_values"]
        feature_values = tf.reshape(feature_values, shape=[batch_size, tf.shape(feature_values)[1], 1])
        feature_fields = features["feature_fields"]
        # feature_fields = tf.reshape(feature_fields, shape=[batch_size, tf.shape(feature_fields)[1], 1])

        ffm_params = FFMParams(self.params)
        all_weights = ffm_params.initialize_weights()
        bias = all_weights["ffm_bias"]
        all_embedding = all_weights["feature_embeddings"]
        weights_first_order = tf.nn.embedding_lookup(all_weights["weights_first_order"], feature_idx)


        ## first_order
        first_order = tf.multiply(weights_first_order, feature_values, name="first_order")
        first_order = tf.reduce_sum(first_order, 2)
        first_order = tf.reduce_sum(first_order, 1, keep_dims=True)

        ## second_order
        second_order = tf.constant(0, dtype=tf.float32)
        # feature_emb = tf.gather(all_embedding, feature_idx, name="feature_emb")
        # print("************** feature_emb= ", feature_emb)
        # quad_term = tf.reduce_sum(feature_emb * tf.transpose(feature_emb, [0, 2, 1, 3]), -1, name="quad_term")  # quad_term:[batch, feature_len, field_size, emb_size]
        # print("************** quad_term= ", quad_term)
        # temp = []
        # for i in range(1, feature_size+1):
        #     temp.append()
        for i in range(1, feature_size+1):
            for j in range(i+1, feature_size+1):
                field_i, field_j = self.field_dict[i], self.field_dict[j]
                emb_i, emb_j = all_embedding[i, field_j, :], all_embedding[j, field_i, :]
                val_i, val_j = origin_feature[:, i], origin_feature[:, j]

                field_emb_sum = tf.multiply(emb_i, emb_j)
                val_sum = tf.multiply(val_i, val_j)

                sum = tf.multiply(tf.reduce_sum(field_emb_sum), tf.cast(val_sum, dtype=tf.float32))
                second_order += sum
        ## final objective function   second_order +
        logits = first_order + bias
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
        export_outputs = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(predictions)}
        # predict 输出
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)
        elif mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=loss, eval_metric_ops=eval_metric_ops)
        elif mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=loss, eval_metric_ops=eval_metric_ops, train_op=train_op)
    def train(self):
        session_config = tf.ConfigProto(log_device_placement=False, device_count={"GPU":0})
        session_config.gpu_options.per_process_gpu_memory_fraction = 0.8
        config = tf.estimator.RunConfig(keep_checkpoint_max=2, log_step_count_steps=500, save_summary_steps=50,
                                        save_checkpoints_steps=50000).replace(session_config=session_config)
        ffm_model = tf.estimator.Estimator(model_fn=self.ffm_model_fn, model_dir="../data/model/ffm/", config=config, params=self.params)
        train_spec = tf.estimator.TrainSpec(input_fn=self.train_input_fn)
        eval_spec = tf.estimator.EvalSpec(input_fn=self.test_input_fn, steps=None, start_delay_secs=1000, throttle_secs=1200)
        tf.estimator.train_and_evaluate(ffm_model, train_spec, eval_spec)
        # ffm_model.train(input_fn=self.train_input_fn, hooks=[tf.train.LoggingTensorHook(["feature_idx",
        #                                                                                   "sigmoid_loss"], every_n_iter=500)])
        # eval
        ffm_model.evaluate(input_fn=self.test_input_fn)
        # predict
        preds = ffm_model.predict(input_fn=self.test_input_fn, predict_keys="prob")
        with open("pred.txt", "w") as fo:
            for prob in preds:
                fo.write("%f\n" % (prob["prob"]))
        # export
        feature_spec = {"feature_ids": tf.placeholder(dtype=tf.int64, shape=[None, self.params["field_size"]], name="feature_idx"),
                        "feature_vals": tf.placeholder(dtype=tf.float32, shape=[None, self.params["field_size"]], name="feature_vals")}
        serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)
        ffm_model.export_saved_model(self.params["model_save_dir"], serving_input_receiver_fn)
    def get_dataset(self):
        userData, itemData, train_rating_info, test_rating_info, user_cols, movie_cols = get1MTrainData(self.data_path)
        feature_dict = {"gender": 0, "age": 0, "occupation": 0, "genres": 1, "year": 1}
        self.params["feature_size"] = len(user_cols) + len(movie_cols)
        self.params["field_size"] = len(set(feature_dict.values()))
        featureIdx, fieldIdx = [], []
        field_dict = dict()
        for col, idx in zip(user_cols+movie_cols, list(range(1, len(user_cols+movie_cols)+1))):
            featureIdx.append(idx)
            fieldIdx.append(feature_dict[col.split("_")[0]])
            field_dict[idx] = feature_dict[col.split("_")[0]]
        self.field_dict = field_dict
        self.fieldTable = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(featureIdx, fieldIdx),
                                                      tf.constant(0, dtype=tf.int32))
        def getTable(data, start_idx, f="user"):
            all_idx, infos, originInfos = [], [], []
            test = 1
            for idx, row in data.iterrows():
                if f == 'item' and test < 2:
                    print("item idx=", idx, " row=", row)
                    test += 1
                all_idx.append(idx)
                col_idx = zip(row, list(range(start_idx, start_idx+len(row))))
                features = list(filter(lambda x: x[0]==1, col_idx))
                infos.append(','.join([str(x[1]) for x in features]))
                originInfos.append(','.join([str(x) for x in row]))
            print("trans table (user/item)data len=", len(all_idx))
            default_value = tf.constant("0", dtype=tf.string)
            table = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(all_idx, infos), default_value)
            originTable = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(all_idx, originInfos), default_value)
            return table, originTable
        userTable, userOriginTable = getTable(userData, start_idx=1, f="user")
        itemTable, itemOriginTable = getTable(itemData, start_idx=len(user_cols)+1, f="item")
        def decode(row):
            userId, itemId, label = tf.cast(row[0], dtype=tf.int32), tf.cast(row[1], dtype=tf.int32), row[2]
            userInfo, itemInfo = userTable.lookup(userId), itemTable.lookup(itemId)
            userOriginInfo, itemOriginInfo = userOriginTable.lookup(userId), itemOriginTable.lookup(itemId)
            user_features = tf.strings.to_number(tf.reshape(tf.sparse.to_dense(tf.strings.split([userInfo], ","),
                                                                               default_value="0"), [-1]),
                                                 out_type=tf.int32)
            item_features = tf.strings.to_number(tf.reshape(tf.sparse.to_dense(tf.strings.split([itemInfo], ","),
                                                                               default_value="0"), [-1]),
                                                 out_type=tf.int32)
            user_origin_features = tf.strings.to_number(tf.reshape(tf.sparse.to_dense(tf.strings.split([userOriginInfo], ","),
                                                                                      default_value="0"), [-1]),
                                                        out_type=tf.int32)
            item_origin_features = tf.strings.to_number(tf.reshape(tf.sparse.to_dense(tf.strings.split([itemOriginInfo], ","),
                                                                                      default_value="0"), [-1]),
                                                        out_type=tf.int32)
            origin_feature = tf.concat([user_origin_features, item_origin_features], axis=0)

            feature_idx = tf.concat([user_features, item_features], axis=0)
            feature_fields = self.fieldTable.lookup(feature_idx)
            feature_values = tf.ones_like(feature_idx, dtype=tf.float32)
            label = tf.divide(tf.cast(label, tf.float32), 5)
            feature_dict = {"feature_idx": feature_idx, "feature_values": feature_values, "feature_fields": feature_fields,
                            "itemId": itemId, "userId": userId, "origin_feature": origin_feature,
                            "user_info": userOriginInfo, "item_info": itemOriginInfo, "user_features": user_features, "item_features": item_features}
            return (feature_dict, label)
        train_dataset = tf.data.Dataset.from_tensor_slices(train_rating_info).map(decode, num_parallel_calls=2).repeat(self.params["epochs"])
        train_dataset = train_dataset.prefetch(self.params["batch_size"]*10) \
            .padded_batch(self.params["batch_size"],
                          padded_shapes=({"feature_idx": [None], "feature_values": [None], "feature_fields": [None],
                                          "itemId": [], "userId": [], "origin_feature": [None],
                                          "user_info":[], "item_info":[], "user_features":[None], "item_features":[None]},
                                         []))
        test_dataset = tf.data.Dataset.from_tensor_slices(test_rating_info).map(decode, num_parallel_calls=2).repeat(self.params["epochs"]).prefetch(self.params["batch_size"]*10) \
            .padded_batch(self.params["batch_size"],
                          padded_shapes=({"feature_idx": [None], "feature_values": [None], "feature_fields": [None],
                                          "itemId": [], "userId": [], "origin_feature": [None],
                                          "user_info":[], "item_info":[], "user_features":[None], "item_features":[None]},
                                         []))
        self.train_dataset, self.test_dataset = train_dataset, test_dataset
    def train_input_fn(self):
        if not hasattr(self, 'train_dataset'):
            self.get_dataset()
        return self.train_dataset
    def test_input_fn(self):
        if not hasattr(self, 'test_dataset'):
            self.get_dataset()
        return self.test_dataset
    def test_run_dataset(self):
        self.get_dataset()
        dataset = self.train_dataset.make_initializable_iterator()
        next_ele = dataset.get_next()
        batch = 1
        with tf.train.MonitoredTrainingSession() as sess:
            sess.run(dataset.initializer)
            while batch <= 2:
                value = sess.run(next_ele)
                print("value=", value)
                batch += 1

def main(_):
    params = {"embedding_size": 6, "feature_size": 0, "field_size": 0, "batch_size": 64, "learning_rate": 0.001,"epochs":200,
              "optimizer": "adam", "data_path": "../../data/ml-1m/"}
    fm = FFMModel(params=params)
    # fm.test_run_dataset()
    fm.train()


if __name__ == '__main__':
    print(tf.__version__)
    tf.app.run()


