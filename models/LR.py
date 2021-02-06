import pandas as pd
import numpy as np
from datetime import datetime
from time import time
import math
import csv

def getTrainData(path):
    movie_info = pd.read_csv(path + "/all_movie_info.csv", sep=",", names=["movieId", "movie_title", "genres", "year", "all_tag"], skiprows=1)
    # user_info = pd.read_csv(path + "/all_users.csv", sep=",", names=["maxTime", "minTime", "tagCount", "userId", "dur_day", "avg_day_movie"], skiprows=1)
    # rating = pd.read_csv(path + "/ratings.csv", names=["userId", "movieId", "rating", "timestamp"], skiprows=1, nrows=1000)

    all_genres = pd.read_csv(path+"/all_genres.csv", sep=",", names=["genres"])["genres"].tolist()
    years = pd.read_csv(path+"/all_year.csv", sep=",", names=["year"])["year"].tolist()
    for genres in all_genres:
        movie_info[genres] = 0
    for year in years:
        movie_info[year] = 0
    movie_info["movie_tag_count"] = 0

    print(movie_info["all_tag"].head(10))
    for idx, row in movie_info.iterrows():
        year = row["year"]
        row[year] = 1
        genresStr = row["genres"].split("|")
        for g in genresStr:
            row[g] = 1
        tags = row["all_tag"].split(":")
        row["movie_tag_count"] = len(tags)

    print(movie_info.head(10))

def get1MTrainData(path):
    user_info = pd.read_csv(path+"/users.dat", header=None, encoding='utf-8', delimiter="::", quoting=csv.QUOTE_NONE,
                            names=["userId", "gender", "age", "occupation", "zipcode"])

    print(user_info.head(10))
    movie_info = pd.read_csv(path+"/movies.dat", header=None, delimiter="::", quoting=csv.QUOTE_NONE,
                            names=["movieId", "title", "genres"])
    print(movie_info.head(50))
    all_genres = pd.read_csv(path + "/all_genres.csv", sep=",", names=["genres"])["genres"].tolist()
    years = pd.read_csv(path + "/all_year.csv", sep=",", names=["year"])["year"].tolist()
    for genres in all_genres:
        movie_info[genres] = 0
    for year in years:
        movie_info[year] = 0

    for idx, row in movie_info.iterrows():
        year = row["year"]
        row[year] = 1
        genresStr = row["genres"].split("|")
        for g in genresStr:
            row[g] = 1

    print(movie_info.head(10))
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
    # train_data = getTrainData(path="../data/ml-25m/")
    train_data = get1MTrainData(path="../data/ml-1m/")
    w = LR(train_data, 20)
    print("LR weight=", w)
    