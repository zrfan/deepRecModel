import pandas as pd
import numpy as np
from datetime import datetime
from time import time
import math

def getTrainData(path):
    movie_info = pd.read_csv(path + "/all_movie_info.csv", sep=",", names=["movieId", "movie_title", "genres", "year", "all_tag"], skiprows=1)
    user_info = pd.read_csv(path + "/all_users.csv", sep=",", names=["maxTime", "minTime", "tagCount", "userId", "dur_day"], skiprows=1)

def LR(trainData, iter):
    w = np.zeros(trainData.shape[1]-1)
    alpha = 0.001
    for j in range(iter):
        for i in range(trainData.shape[0]):
            xi = trainData[i][0:-1]
            yi = trainData[i][-1]
            wx = np.dot(w, xi)
            w += alpha * xi * (yi - (np.exp(wx)/(1+np.exp(wx))))
    return w



if __name__ == '__main__':
    train_data = getTrainData(path="../data/ml-25m/")
    LR(train_data, 20)