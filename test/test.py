# -*- coding: utf-8 -*
# tf1.14
import tensorflow as tf
import pandas as pd

def decode(row):
    seq_tags = row[0]
    all_tags = tf.sparse.to_dense(tf.string_split([seq_tags], ","), default_value="0")
    all_tags = tf.reshape(all_tags, [-1])

    split_tags = tf.sparse.to_dense(tf.string_split(all_tags, "|"), default_value="0")

    res = tf.strings.to_number(split_tags, out_type=tf.int32)
    return {"seq_tags": seq_tags, "all_tags": all_tags, "split_tags": split_tags, "res": res}  # ,
def test_run_dataset():
    data = pd.DataFrame(['12,13,14|15,10|11|12',
                         '12,11,7|8',
                         '1,2,3,4,7',
                         '1|4,5,6'], columns=["seq_tags"])
    print(data)
    dataset = tf.data.Dataset.from_tensor_slices(data).map(decode).padded_batch(2, padded_shapes=({"seq_tags": [],
                                                                                                   "all_tags": [None],
                                                                                                   "split_tags": [None, None],
                                                                                                   "res": [None, None]}))
    dataset = dataset.make_initializable_iterator()
    data = dataset.get_next()

    batch = 1
    with tf.train.MonitoredTrainingSession() as sess:
        sess.run(dataset.initializer)
        while batch <= 2:
            value = sess.run([data])
            print("value=", value)
            batch += 1

def main(_):
    test_run_dataset()


if __name__ == '__main__':
    print(tf.__version__)
    tf.app.run()