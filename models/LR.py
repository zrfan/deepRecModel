#-*- coding: utf-8 -*
import pandas as pd
import numpy as np
from datetime import datetime
from time import time
import math
import csv
from data_process import getYear

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
    for g in genderList:
        user_info["gender_"+str(g)] = 0
    for a in ageList:
        user_info["age_"+str(a)] = 0
    for o in occupationList:
        user_info["occ_"+str(o)] = 0
    for idx, row in user_info.iterrows():
        gender, age, occ = str(row["gender"]), str(row["age"]), str(row["occupation"])
        row["gender_"+gender], row["age_"+age], row["occ_"+occ] = 1, 1, 1

    user_info = user_info[user_cols]
    print(user_info.head(10))
    movie_info = pd.read_csv(path+"/movies.dat", header=None, delimiter="::", quoting=csv.QUOTE_NONE, names=["movieId", "title", "genres"])
    movie_info["year"] = movie_info["title"].apply(lambda x: getYear(x))
    all_genres = pd.read_csv(path + "/../all_genres.csv", sep=",", names=["genres"])["genres"].tolist()
    years = pd.read_csv(path + "..//all_year.csv", sep=",", names=["year"])["year"].tolist()
    for genres in all_genres:
        movie_info["generes_"+genres] = 0
    for year in years:
        movie_info["year_"+str(year)] = 0

    for idx, row in movie_info.iterrows():
        year, genresStr = str(row["year"]), row["genres"].split("|")
        row["year_"+year] = 1
        for g in genresStr:
            row["generes_"+g] = 1
    movie_cols = ["generes_"+x for x in all_genres]+["year_"+str(x) for x in years]+["movieId"]
    print("movie_cols=", movie_cols)
    movie_info = movie_info[movie_cols]
    print(movie_info.head(10))
    rating_info = pd.read_csv(path+"/ratings.dat", header=None, delimiter="::", quoting=csv.QUOTE_NONE, names=["userId", "movieId", "ratings", "timestamp"])
    print(rating_info.head(10))

    return user_info.set_index("userId").to_dict(), movie_info.set_index("movieId").to_dict(), rating_info, user_cols, movie_cols
def LR(userData, itemData, clickData, user_cols, movie_cols, iter):
    print("uid=1", userData.keys())
    w = np.zeros(len(user_cols)+len(movie_cols)-1)
    alpha = 0.001
    for j in range(iter):
        for idx, row in clickData.iterrows():
            print("row=", row)
            trainData = userData.get(row["userId"]) + itemData.get(row["movieId"])
            print("trainData=", trainData)
            xi = trainData[0:-1]
            yi = float(row["ratings"])/5
            wx = np.dot(w, xi)
            w += alpha * xi * (yi - (np.exp(wx)/(1+np.exp(wx))))
    return w



if __name__ == '__main__':
    # train_data = getTrainData(path="../data/ml-25m/")
    user_info, movie_info, rating_info, user_cols, movie_cols = get1MTrainData(path="../data/ml-1m/")
    w = LR(user_info, movie_info, rating_info,user_cols, movie_cols, 20)
    print("LR weight=", w)
