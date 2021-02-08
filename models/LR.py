#-*- coding: utf-8 -*
import pandas as pd
import numpy as np
from datetime import datetime
from time import time
import math
import csv
# from data_process import getYear

def getYear(x):
    arr = x.split("(")
    if len(arr) == 1:
        return 0
    y = 0
    for t in arr:
        try:
            y = int(t[0:4])
        except:
            continue
    return y

def getTrainData(path):
    movie_info = pd.read_csv(path + "/all_movie_info.csv", sep=",", names=["movieId", "movie_title", "genres", "year", "all_tag"], skiprows=1)
    # user_info = pd.read_csv(path + "/all_users.csv", sep=",", names=["maxTime", "minTime", "tagCount", "userId", "dur_day", "avg_day_movie"], skiprows=1)
    # rating = pd.read_csv(path + "/ratings.csv", names=["userId", "movieId", "rating", "timestamp"], skiprows=1, nrows=1000)

    all_genres = pd.read_csv(path+"/../all_genres.csv", sep=",", names=["genres"])["genres"].tolist()
    years = pd.read_csv(path+"/../all_year.csv", sep=",", names=["year"])["year"].tolist()
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
    genderList = ["F", "M"]
    ageList = [1, 18, 25, 35, 45, 50, 56]
    occupationList = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    user_cols = ["gender_"+str(x) for x in genderList] + ["age_"+str(x) for x in ageList] + ["occ_"+str(x) for x in occupationList] + ["userId"]
    genderInfo = pd.get_dummies(user_info["gender"], sparse=True)
    ageInfo = pd.get_dummies(user_info["age"], sparse=True)
    occInfo = pd.get_dummies(user_info["occupation"], sparse=True)
    occInfo.columns =["occ_"+str(x) for x in occupationList]
    ageInfo.columns = ["age_"+str(x) for x in ageList]
    genderInfo.columns = ["gender_"+x for x in genderList]
    user_info = user_info.join(genderInfo).join(ageInfo).join(occInfo)[user_cols]
    user_info = user_info.set_index("userId")
    print(user_info.head(10))
    userDict = {}

    movie_info = pd.read_csv(path+"/movies.dat", header=None, delimiter="::", quoting=csv.QUOTE_NONE, names=["movieId", "title", "genres"])
    movie_info["year"] = movie_info["title"].apply(lambda x: getYear(x))
    genresList = pd.read_csv(path + "/../all_genres.csv", sep=",", names=["genres"])["genres"].tolist()
    yearList = pd.read_csv(path + "..//all_year.csv", sep=",", names=["year"])["year"].tolist()
    movie_cols = ["generes_" + x for x in genresList] + ["year_" + str(x) for x in yearList] + ["movieId"]
    yearInfo = pd.get_dummies(movie_info["year"], sparse=True)
    yearInfo.columns = ["year_"+str(x) for x in yearInfo.columns ]
    print(yearInfo.head(10))
    movie_info = movie_info.join(yearInfo)
    print(movie_info.head(10))
    movie_info = movie_info.set_index("movieId")
    print(movie_info.head(10))
    for g in genresList:
        movie_info["genres_"+g] = 0
    for idx, row in movie_info.iterrows():
        gList = row["genres"].split("|")
        for g in gList:
            movie_info.loc[idx, "genres_"+g] = 1
    movie_info = movie_info[movie_cols]
    print(movie_info.head(10))

    rating_info = pd.read_csv(path+"/ratings.dat", header=None, delimiter="::", quoting=csv.QUOTE_NONE, names=["userId", "movieId", "ratings", "timestamp"])
    
    return user_info.values, movie_info.values, rating_info, user_cols, movie_cols
def LR(userData, itemData, clickData, user_cols, movie_cols, iter):
    # print("uid=1", userData.keys())
    w = np.zeros(len(user_cols)+len(movie_cols)-2)
    alpha = 0.001
    for j in range(iter):
        for idx, row in clickData.iterrows():
            print("row=", row, " user_info:", userData[int(row["userId"])-1], )
            trainData = userData[int(row["userId"])-1].tolist() + itemData[int(row["movieId"])-1].tolist()
            print("trainData=", trainData)
            xi = trainData
            yi = float(row["ratings"])/5
            print("w=", w)
            wx = np.dot(w, xi)
            print("wx=", wx)
            w += alpha * xi * (yi - (np.exp(wx)/(1+np.exp(wx))))
    return w

if __name__ == '__main__':
    # train_data = getTrainData(path="../data/ml-25m/")
    user_info, movie_info, rating_info, user_cols, movie_cols = get1MTrainData(path="../data/ml-1m/")
    w = LR(user_info, movie_info, rating_info,user_cols, movie_cols, 20)
    print("LR weight=", w)
