#-*- coding: utf-8 -*
# tf1.13
from __future__ import division
from math import exp
from numpy import *
from random import normalvariate  # 正态分布
from sklearn import preprocessing
import numpy as np
from data_util import get1MTrainData

class FM(object):
    def __init__(self, data_path, iter, task_type, k=8, alpha=0.001):
        self.data_path, self.feature_potential, self.iter = data_path, k, iter
        self.task_type = task_type  # 0: classification 1: regression
        self.alpha, self._w, self._w_0, self.v = alpha, None, None, None
        self.with_col, self.first_col = None, None
    def sigmoid(self, inx):
        return 1.0 / (1 + exp(-inx))
    def fit(self):
        userData, itemData, rating_info, user_cols, movie_cols = get1MTrainData(self.data_path)
        k = self.feature_potential
        m, n = len(rating_info), len(user_cols)+len(movie_cols)-2
        # 初始化参数
        w = np.zeros((n, 1))  # n是特征的个数
        w_0 = 0
        v = normalvariate(0, 0.2) * np.ones((n, k))
        # v = np.random.normal(size=(n, k))
        for it in range(self.iter):
            for idx, row  in rating_info.iterrows():
                userId, itemId = int(row["userId"]), int(row["movieId"])
                userInfo, movieInfo = userData.loc[userId, :], itemData.loc[itemId, :]
                trainData = userInfo.tolist() + movieInfo.tolist()
                # x = np.mat(trainData)
                x = np.asarray(trainData)[np.newaxis, :]
                # x = np.asarray(trainData)[:, np.newaxis]
                print("x shape", x.shape)
                y = float(row["ratings"])/5
                # 对应点积的地方通常会有sum，对应位置积的地方通常没有
                # FM的二阶项：1/2 \sum_{f=1}^k ((\sum_{i=1}^n v_{i,f}x_i)^2 - \sum_{i=1}^n v_{i,f}^2 * x_i^2)
                inter_sum = np.dot(x, v) # x * v  # xi * vi, xi与vi的矩阵点积  shape=(1, 8)
                print("v=", v)
                print("inter_sum=", inter_sum)
                # xi与xi的对应位置乘积 与 xi^2与vi^2对应位置的乘积的点积，
                inter_sum_sqr = np.multiply(x, x) * np.multiply(v, v)    # multiply对应元素相乘
                # inter_sum_sqr = inter_sum_sqr.sum(axis=0)    # shape=(1, 8)
                print("inter_sum_sqr=", inter_sum_sqr)
                # 交叉项 xi*vi*xi*vi - xi^2*vi^2
                interaction = np.sum(np.multiply(inter_sum, inter_sum) - inter_sum_sqr) / 2
                print("interaction=", interaction)
                # 计算预测的输出
                p = w_0 + x*w + interaction
                print("p=", p)
                # 计算sigmoid（y*pred_y）-1
                loss = (self.sigmoid(y * p[0, 0]) -1)*y if self.task_type == 0 else self.sigmoid(p[0,0])-y
                # 更新参数
                w_0 = w_0 - self.alpha * loss
                for i in range(n):
                    if x[0, i] != 0:
                        w[i, 0] = w[i, 0] - self.alpha * loss * x[0, i]
                        for j in range(k):
                            v[i, j] = v[i, j] - self.alpha * loss * (x[0, i] * inter_sum[0, j] - v[i,j]*x[0, i]*x[0, i])
                print("w=", w)
        
        self._w_0, self._w, self._v = w_0, w, w
    def predict(self, data):
        w_0, w, v = self._w_0, self._w, self._v
        m, n = shape(data)
        result = []
        for x in range(m):
            inter_1 = np.mat(X[x]) * v
            inter_2 = np.mat(np.multiply(X[x], X[x])) * np.multiply(v, v) # multiply对应元素相乘
            # 完成交叉项
            interaction = np.sum(np.multiply(inter_1, inter_2) - inter_2) / 2
            p = w_0 + X[x] * w + interaction # 计算预测的输出
            pre = self.sigmoid(p[0, 0])
            result.append(pre)
        return result

    # def loadDataSet(self, data, with_col=True, first_col=2):

if __name__=='__main__':
    fm = FM(data_path="../data/ml-1m/", iter=10, task_type=1)
    fm.fit()

