#-*- coding: utf-8 -*
import pandas as pd
import numpy as np
from datetime import datetime
from time import time
import math
import csv
from data_util import get1MTrainData

def LR(userData, itemData, clickData, user_cols, movie_cols, iter):
    w = np.zeros(len(user_cols)+len(movie_cols)-2)
    alpha = 0.001
    for j in range(iter):
        for idx, row in clickData.iterrows():
            userId, itemId = int(row["userId"]), int(row["movieId"])
            userInfo, movieInfo = userData.loc[userId, :], itemData.loc[itemId, :]
            trainData = userInfo.tolist() + movieInfo.tolist()
            # print("trainData=", trainData)
            xi = np.asarray(trainData)
            yi = float(row["ratings"])/5
            print("w=", w)
            wx = np.dot(w, xi)
            print("wx=", wx)
            w += alpha * xi * (yi - (np.exp(wx)/(1+np.exp(wx))))
    return w

if __name__ == '__main__':
    # train_data = getTrainData(path="../data/ml-25m/")
    user_info, movie_info, rating_info, user_cols, movie_cols = get1MTrainData(path="../data/ml-1m/")
    w = LR(user_info, movie_info, rating_info,user_cols, movie_cols, 200)
    print("LR weight=", w)
