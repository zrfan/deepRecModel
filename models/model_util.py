# -*- coding: utf-8 -*
# tf1.14
import tensorflow as tf

# register user / item feature table
def registerHashTable(data, start_feature_idx, flag="user"):
    all_idx, infos, originInfos = [], [], []
    default_value = tf.constant("0", dtype=tf.string)
    for idx, row in data.iterrows():
        all_idx.append(idx)
        # feature_value and feature_idx
        col_idx = zip(row, list(range(start_feature_idx, start_feature_idx+len(row))))
        features = list(filter(lambda x: x[0]==1, col_idx))
        infos.append(','.join([str(x[1]) for x in features]))
        originInfos.append(','.join([str(x) for x in row]))
    infoTable = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(all_idx, infos), default_value)
    originInfoTable = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(all_idx, originInfos), default_value)
    return infoTable, originInfoTable

def registerAllFeatureIdxHashTable(userData, itemData):
    all_idx, infos, originInfos = [], [], []
    default_value = tf.constant("0", dtype=tf.string)
    for uidx, row in userData.iterrows():
        all_idx.append(uidx)
        col_idx = zip(row, list(range(0, len(row))))
        features = list(filter(lambda x: x[0]==1, col_idx))
        infos.append(','.join([str(x[1]) for x in features]))
        originInfos.append(','.join([str(x) for x in row]))
    ulen, user_feature_len = len(all_idx), len(userData.columns)
    for midx, row in itemData.iterrows():
        all_idx.append(midx+ulen)
        col_idx = zip(row, list(range(user_feature_len, user_feature_len+len(row))))
        features = list(filter(lambda x: x[0]==1, col_idx))
        infos.append(','.join([str(x[1]) for x in features]))
        originInfos.append(','.join([str(x) for x in row]))
    allInfoTable = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(all_idx, infos), default_value)
    originInfoTable = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(all_idx, originInfos), default_value)
    return allInfoTable, originInfoTable

def registerAllFeatureHashTable(userData, itemData):
    all_idx, originInfos = [], []
    default_value = tf.constant("0", dtype=tf.string)
    for uidx, row in userData.iterrows():
        all_idx.append(uidx)
        originInfos.append(','.join([str(x) for x in row]))
    ulen, user_feature_len = len(all_idx), len(userData.columns)
    for midx, row in itemData.iterrows():
        all_idx.append(midx + ulen)
        originInfos.append(','.join([str(x) for x in row]))
    originInfoTable = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(all_idx, originInfos),
                                                  default_value)
    return originInfoTable