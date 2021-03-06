#!/bin/sh


# movielens ml-25m dataset
wget -c http://files.grouplens.org/datasets/movielens/ml-25m.zip
wget -c http://files.grouplens.org/datasets/movielens/ml-1m.zip
python data_process.py
python 1m_data_process.py

# criteo数据集
wget --no-check-certificate https://s3-eu-west-1.amazonaws.com/criteo-labs/dac.tar.gz



# ESSM public dataset, Alibaba click and conversion prediction
# https://tianchi.aliyun.com/dataset/dataDetail?dataId=408&userId=1
# train
wget -c https://jupter-oss.oss-cn-hangzhou.aliyuncs.com/file/opensearch/documents/408/sample_train.tar.gz?Expires=1612265517&OSSAccessKeyId=LTAI4GGBCQcb7KD7NwKinA3D&Signature=RxIDw519awjeR8ArtnZqBvjzScw%3D&response-content-disposition=attachment%3B%20
# test
wget -c https://jupter-oss.oss-cn-hangzhou.aliyuncs.com/file/opensearch/documents/408/sample_test.tar.gz?Expires=1612265650&OSSAccessKeyId=LTAI4GGBCQcb7KD7NwKinA3D&Signature=ZuGp4sUMIKZAZkZhFu87s96Csd8%3D&response-content-disposition=attachment%3B%20

# download albert_tiny model
wget -c https://storage.googleapis.com/albert_zh/albert_tiny_489k.zip


# spark-nlp albert_baes
wget -c https://storage.googleapis.com/tfhub-modules/google/albert_base/3.tar.gz

# spark-nlp albert_large
wget -c https://storage.googleapis.com/tfhub-modules/google/albert_large/3.tar.gz

# Amazon data
#https://github.com/zhougr1993/DeepInterestNetwork/tree/master/utils
wget -c http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz
gzip -d reviews_Electronics_5.json.gz
wget -c http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Electronics.json.gz
gzip -d meta_Electronics.json.gz
