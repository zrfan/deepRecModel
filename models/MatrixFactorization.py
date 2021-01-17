import pandas as pd
import numpy as np
from datetime import datetime
from time import time
import math

class LFM(object):
    def __init__(self, k=10, iter=10, alpha=0.001, neg_ratio=0.1, lamda=0.1, dataPath="../data/ml-25m/"):
        self.userCount = None
        self.itemCount = None
        self.itemPool = None
        self.user_matrix = None
        self.item_matrix = None
        self.train_user_items = None
        self.test_user_items = None
        self.shape = None
        self.rmse = None
        self.k = k
        self.iter = iter
        self.alpha = alpha
        self.neg_ratio = neg_ratio
        self.lamda = lamda
        self.process_data(path)
        self.initUserItemMatrix()
    def process_data(self, path):
        userRating = pd.read_csv(path + "/ratings.csv",
                                 names=["userId", "movieId", "rating", "timestamp"], sep=",", skiprows=1)
        print("userRating=\n", userRating.head(10))
        userCount = max(userRating["userId"]) + 1
        movieCount = max(userRating["movieId"]) + 1

        print("userCount=", userCount, " movieCount=", movieCount)
        self.userCount = userCount
        self.itemCount = movieCount
        groupUserItem = userRating.groupby("userId").apply(lambda x: x.sort_values("timestamp", ascending=True))
        self.train_user_items = dict()
        self.test_user_items = dict()
        for user, group in groupUserItem:
            self.train_user_items[user] = set(group['movieId'][:-2])
            self.test_user_items[user] = group["movieId"][-2:]
        print("train_user_items count=", self.train_user_items.__len__(),
              " test_user_items count=", self.test_user_items.__len__())

        itemCount = userRating.groupby("movieId").size()
        self.itemPool = dict(itemCount)

    def randSelectNegSample(self, userId, items):
        userRate = dict()
        for item in items:
            userRate[item] = 1
        negNum = int(round(len(items)*self.neg_ratio))
        N = 0
        # allItemSet = self.itemPool.keys(
        for item in self.itemPool.keys():
            if N > negNum:
                break
            if item in items:
                continue
            N += 1
            userRate[item] = 0
        return userRate
    def lfmPredict(self, userId, itemId):
        userVec = np.mat(self.user_matrix.iloc[userId, :])
        itemVec = np.mat(self.item_matrix.iloc[itemId, :])

        rating = (userVec * itemVec.T).sum()
        # print("userVec=", userVec, " itemVec=", itemVec, " rating=", rating)
        return 1.0 / (1 + np.exp(-rating))

    def SGD(self):
        alpha = self.alpha
        i = 0
        for _ in range(self.iter):
            for userId, items in self.train_user_items.items():
                i += 1
                if i % 100 == 0:
                    print("i=", i)
                userRating = self.randSelectNegSample(userId, items)
                for itemId, rating in userRating.items():
                    dRate = rating - self.lfmPredict(userId, itemId)
                    # loss = (u_i * m_j - rating)^2 + \lambda (u_i^2 + m_j^2)
                    for f in range(0, self.k):
                        self.user_matrix.iloc[userId, f] += alpha * (dRate * self.item_matrix.iloc[itemId, f] -
                                                              self.lamda * self.user_matrix.iloc[userId, f])
                        self.item_matrix.iloc[itemId, f] += alpha * (dRate * self.user_matrix.iloc[userId, f] -
                                                              self.lamda * self.item_matrix.iloc[itemId, f])
                    # print("userVec=", self.user_matrix.iloc[userId, :], " itemVec=", self.item_matrix.iloc[itemId, :])
            # alpha *= 0.9
            # print("alpha=", alpha)
    def test(self):
        correct, all = 0, 0
        for userId, items in self.test_user_items.items():
            for itemId in items:
                predict = self.lfmPredict(userId, itemId)
                all += 1
                if predict > 0.5:
                    correct += 1
        print("accuracy=", correct*1.0/all)

    def train(self):
        self.SGD()
        self.test()


    def initUserItemMatrix(self):
        user = np.random.rand(self.userCount, self.k)
        item = np.random.rand(self.itemCount, self.k)
        self.user_matrix = pd.DataFrame(user, columns=range(0, self.k))
        self.item_matrix = pd.DataFrame(item, columns=range(0, self.k))


if __name__ == '__main__':
    path = "../data/ml-25m/"
    lfm = LFM(k=8, iter=1)
    lfm.train()
    userId = 3412
    itemId = [2341, 3452, 3435, 3241]
    rating = list(map(lambda item: lfm.lfmPredict(userId, item), itemId))
    print("Result Rating=", rating)



