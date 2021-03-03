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
    print(user_info.shape)
    genderList = ["F", "M"]
    ageList = [1, 18, 25, 35, 45, 50, 56]
    occupationList = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    zipcodeList = list(set(user_info["zipcode"].tolist()))
    # print("zipcodeList=", zipcodeList)
    print("zipcode len=", len(zipcodeList))
    user_cols = ["gender_"+str(x) for x in genderList] + ["age_"+str(x) for x in ageList] + ["occupation_"+str(x) for x in occupationList]
    genderInfo = pd.get_dummies(user_info["gender"], sparse=True)
    ageInfo = pd.get_dummies(user_info["age"], sparse=True)
    occInfo = pd.get_dummies(user_info["occupation"], sparse=True)
    occInfo.columns =["occupation_"+str(x) for x in occupationList]
    ageInfo.columns = ["age_"+str(x) for x in ageList]
    genderInfo.columns = ["gender_"+x for x in genderList]
    user_info = user_info.join(genderInfo).join(ageInfo).join(occInfo)[user_cols+["userId"]]
    user_info = user_info.set_index("userId")
    # take1 = user_info.head(1)
    print("user columns len=", len(user_cols), user_info.columns)
    print("take1 user info= ")
    # print(take1)
    # for x in take1:
    #     print(x, " ", take1[x])

    movie_info = pd.read_csv(path+"/movies.dat", header=None, delimiter="::", quoting=csv.QUOTE_NONE, names=["movieId", "title", "genres"])
    movie_info["year"] = movie_info["title"].apply(lambda x: getYear(x))
    genresList = list(set(pd.read_csv(path + "/../all_genres.csv", sep=",", names=["genres"])["genres"].tolist()))
    yearList = list(set(movie_info["year"].tolist()))
    print("genres len=", len(genresList), " years len=", len(yearList))
    
    yearInfo = pd.get_dummies(movie_info["year"], sparse=True)
    print(yearInfo.head(1))
    yearInfo.columns = ["year_"+str(x) for x in yearInfo.columns ]
    movie_info = movie_info.join(yearInfo)

    for g in genresList:
        movie_info["genres_"+g] = 0
    for idx, row in movie_info.iterrows():
        gList = row["genres"].split("|")
        for g in gList:
            movie_info.loc[idx, "genres_"+g] = 1
    movie_cols = ["genres_" + x for x in genresList] + ["year_" + str(x) for x in yearList]
    movie_info = movie_info[movie_cols+["movieId"]]
    movie_info = movie_info.set_index("movieId")
    take1 = movie_info.head(1)
    print("movie columns len=", len(movie_cols), movie_info.columns)
    print("take1 movie info= ")
    for x in take1:
        print(" ", x)

    train_rating_info = pd.read_csv(path+"/train_rating.dat", header=None, delimiter=",", quoting=csv.QUOTE_NONE, names=["userId", "movieId", "ratings", "timestamp"])
    test_rating_info = pd.read_csv(path+"/test_rating.dat", header=None, delimiter=",", quoting=csv.QUOTE_NONE, names=["userId", "movieId", "ratings", "timestamp"])

    return user_info, movie_info, train_rating_info, test_rating_info, user_cols, movie_cols

def splitTrainAndTestRating(path):
    rating_info = pd.read_csv(path+"/ratings.dat", header=None, delimiter="::", quoting=csv.QUOTE_NONE, names=["userId", "movieId", "ratings", "timestamp"])
    rating_info = rating_info.sort_values(by="timestamp", ascending=True)

    size = int(rating_info.shape[0]*0.9)
    train_rating = rating_info[:size]
    test_rating = rating_info[size:]
    train_rating.to_csv(path+"/train_rating.dat", index=False, header=None, sep=",", quoting=csv.QUOTE_NONE)
    test_rating.to_csv(path+"/test_rating.dat", index=False, header=None, sep=",", quoting=csv.QUOTE_NONE)

def get1MTrainDataOriginFeatures(path):
    user_info = pd.read_csv(path+"/users.dat", header=None, encoding='utf-8', delimiter="::", quoting=csv.QUOTE_NONE,
                            names=["userId", "gender", "age", "occupation", "zipcode"])
    print(user_info.shape)
    genderList = ["F", "M"]
    ageList = [1, 18, 25, 35, 45, 50, 56]
    occupationList = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    zipcodeList = list(set(user_info["zipcode"].tolist()))
    # print("zipcodeList=", zipcodeList)
    print("zipcode len=", len(zipcodeList))
    user_info = user_info.set_index("userId")

    movie_info = pd.read_csv(path+"/movies.dat", header=None, delimiter="::", quoting=csv.QUOTE_NONE, names=["movieId", "title", "genres"])
    movie_info["year"] = movie_info["title"].apply(lambda x: getYear(x))
    genresList = list(set(pd.read_csv(path + "/../all_genres.csv", sep=",", names=["genres"])["genres"].tolist()))
    yearList = list(set(movie_info["year"].tolist()))
    print("genres len=", len(genresList), " years len=", len(yearList))

    movie_info = movie_info["movieId", "genres", "year"]
    movie_info = movie_info.set_index("movieId")

    train_rating_info = pd.read_csv(path+"/train_rating.dat", header=None, delimiter=",", quoting=csv.QUOTE_NONE, names=["userId", "movieId", "ratings", "timestamp"])
    test_rating_info = pd.read_csv(path+"/test_rating.dat", header=None, delimiter=",", quoting=csv.QUOTE_NONE, names=["userId", "movieId", "ratings", "timestamp"])

    return user_info, movie_info, train_rating_info, test_rating_info, user_info.columns,movie_info.columns

def get1MTrainDataWithNeg(path):
    user_info, movie_info, rating_info, user_cols, movie_cols = get1MTrainData(path)

def main():
    path = "../data/ml-1m/"
    # get1MTrainData(path)
    splitTrainAndTestRating(path)

if __name__=="__main__":
    main()