import pandas as pd
import numpy as np
from datetime import datetime
import time
import math

def mergeTag(df, column="all_tag"):
    tags = list(df[column])
    return ':'.join(tags)
def mergeTagName(df, column="tag_name"):
    names = list(df[column])
    return ":".join(names)
def addGenres(all_genres, x):
    for t in x.split["|"]:
        all_genres.add(t)

def getAllMovieTagInfo(path):
    genome_movie_tag_score = pd.read_csv(path + "/genome-scores.csv",
                                         names=["movieId", "tagId", "tag_relevance"],
                                         skiprows=1)
    # .sort_values(by=['movieId', 'tag_relevance'], ascending=[True, False])
    genome_tag_info = pd.read_csv(path + "/genome-tags.csv", names=["tagId", "tag_name"], skiprows=1).sort_values(by=["tagId"], ascending=True)
    print("genome_movie_tag_score size=", genome_movie_tag_score.size,
         " genome_taged_movieId_cnt=", genome_movie_tag_score["movieId"].drop_duplicates().size,
          " genome_tag_info count=\n", genome_tag_info.count(),
          " all_genome_tagId_cnt=", genome_tag_info["tagId"].drop_duplicates().size)

    print("genome_movie_tag_score count=\n", genome_movie_tag_score.count())
    genome_tag_info = pd.merge(genome_movie_tag_score, genome_tag_info, how='left', on=['tagId'])  #[movieId, tagId, tag_relevance, tag_name]
    print("merged genome_movie_tag_info count=\n", genome_tag_info.count(), " genome_taged_movieId_cnt=",
          genome_tag_info["movieId"].drop_duplicates().size, " all genome_tagId_cnt=", genome_tag_info["tagId"].drop_duplicates().size)
    return genome_tag_info
def cleanStr(x):
    try:
        s = str(eval(x))
    except:
        print("s=", x)
        s = x
    return s
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

def getMovieInfo(path):
    movie_info = pd.read_csv(path + "/movies.csv", names=["movieId", "movie_title", "genres"], skiprows=1)
    movie_info["year"] = movie_info["movie_title"].apply(lambda x: getYear(x))
    # movie_info["year"] = movie_info["year"].apply(lambda x: int(x))
    print(movie_info["year"].head(10))
    all_year = list(set(movie_info["year"].tolist()))
    print("all_year=", all_year)
    all_year = pd.DataFrame({"year": all_year})
    print(all_year.head(10))
    all_year.to_csv(path+"/all_year.csv", index=False, header=False)

    movie_info["movie_title"] = movie_info["movie_title"].apply(lambda x: " ".join(x.split(" ")[0:-1]))
    all_genres = []
    for g in movie_info["genres"].tolist():
        for x in g.split("|"):
            all_genres.append(x)
    all_genres = list(set(all_genres))
    print("all_genres=", all_genres)
    print("genres len=", len(movie_info["genres"].tolist()), "all_genres=", len(all_genres))
    all_genres = pd.DataFrame({"genres": all_genres})
    all_genres.to_csv(path + "/all_genres.csv", index=False, header=False)
    return movie_info
def getUserTagMovie(path):  # [movieId, tagId, tag_relevance, tag_name]
     # all_movie_tag_info = getAllMovieTagInfo()
     userTagMovie = pd.read_csv(path + "/tags.csv", sep=',', names=["userId", "movieId", "tag_name", "timestamp"], skiprows=1)
     userTagMovie["date"] = userTagMovie["timestamp"].apply(lambda x: time.localtime(x))
     userTagMovie["year"] = userTagMovie["date"].apply(lambda x: x[0])
     userTagMovie["month"] = userTagMovie["date"].apply(lambda x: x[1])
     print(userTagMovie.head(10))
     print("user_tag_movie count=\n", userTagMovie.count(), " max time=", userTagMovie.max(),
           " min time=", userTagMovie.min())
     userMaxTime = userTagMovie.groupby("userId")["timestamp"].max()
     print(userMaxTime.head(10))
     userMinTime = userTagMovie.groupby("userId")["timestamp"].min()

     userTagCount = userTagMovie.groupby("userId")['timestamp'].agg({'tagCount': np.size, "maxTime": np.max, "minTime": np.min})
     print("all agg=\n", userTagCount.head(10))
     userInfo = pd.DataFrame({"userId": userTagCount.index, "tagCount": userTagCount['tagCount'],
                              "maxTime": userTagCount["maxTime"], "minTime": userTagCount["minTime"]})
     userInfo["dur_day"] = (userInfo["maxTime"] - userInfo["minTime"]) / 60 / 60 / 24
     userInfo["avg_day_movie"] = userInfo["dur_day"] / userInfo["tagCount"]

     print("user_info=\n", userInfo.head(10))
     userInfo.to_csv(path+"/all_users.csv", index=False, header=True)



def generate_movie_data(path):
    # 处理训练数据
    movie_info = getMovieInfo(path)
    tag_info = getAllMovieTagInfo(path)

    user_rating_movie = pd.read_csv(path+"/ratings.csv", sep=',', nrows=10000, names=["userId", "movieId", "rating", "timestamp"], skiprows=1)


    all_movieId_cnt = movie_info["movieId"].drop_duplicates().size
    rating_movieId_cnt = user_rating_movie["movieId"].drop_duplicates().size
    print("all movie_info size=", movie_info.size, " all_movieId_cnt=", all_movieId_cnt, " rating_movieId_cnt=", rating_movieId_cnt)


    tag_info["all_tag"] = tag_info['tagId'].map(str) + '|' + tag_info["tag_relevance"].map(str) + "|" + tag_info["tag_name"]
    tag_info = tag_info[["movieId", "all_tag"]]
    print("movie_tag_info before group size=", tag_info.count(), "\n", tag_info.head(10))
    movie_tag_info = tag_info.groupby('movieId').apply(mergeTag, column="all_tag")
    movie_tag_info = pd.DataFrame({'movieId': movie_tag_info.index, "all_tag": movie_tag_info.values})

    print("grouped movie_tag_info=", movie_tag_info.count(), "\n", movie_tag_info.head(10))

    movie_info = pd.merge(movie_info, movie_tag_info, how='left', on=['movieId'])   # [movieId, movie_title, genres, year, all_tag]
    print("movie_info=\n", movie_info.count(), " movieId cnt=", movie_info["movieId"].drop_duplicates().size)
    print("movie_info=\n", movie_info.head(10))

    movie_info.to_csv(path + "/all_movie_info.csv", index=False)

# all_genres= ['Sci-Fi', 'Musical', 'Thriller', 'War', 'Romance', 'Animation', 'Comedy', 'Crime', 'IMAX', 'Documentary', 'Western', 'Drama', 'Film-Noir', 'Fantasy', 'Adventure', 'Horror', '(no genres listed)', 'Children', 'Mystery', 'Action']
# genres len= 62423 all_genres= 20
# genome_movie_tag_score size= 46753344  genome_taged_movieId_cnt= 13816  genome_tag_info count=
#  tagId       1128
# tag_name    1128
# dtype: int64  all_genome_tagId_cnt= 1128
# genome_movie_tag_score count=
#  movieId          15584448
# tagId            15584448
# tag_relevance    15584448
# dtype: int64
# merged genome_movie_tag_info count=
#  movieId          15584448
# tagId            15584448
# tag_relevance    15584448
# tag_name         15584448
# dtype: int64  genome_taged_movieId_cnt= 13816  all genome_tagId_cnt= 1128
# all movie_info size= 249692  all_movieId_cnt= 62423  rating_movieId_cnt= 3287
# movie_tag_info before group size= movieId    15584448
# all_tag    15584448
# dtype: int64
#     movieId                              all_tag
# 0        1                        1|0.02875|007
# 1        1  2|0.023749999999999997|007 (series)
# 2        1                3|0.0625|18th century
# 3        1          4|0.07574999999999997|1920s
# 4        1                      5|0.14075|1930s
# 5        1                      6|0.14675|1950s
# 6        1                       7|0.0635|1960s
# 7        1                      8|0.20375|1970s
# 8        1          9|0.20199999999999999|1980s
# 9        1              10|0.03075|19th century
# grouped movie_tag_info= all_tag    13816
# movieId    13816
# dtype: int64
#                                               all_tag  movieId
# 0  1|0.02875|007,2|0.023749999999999997|007 (seri...        1
# 1  1|0.041250000000000016|007,2|0.040499999999999...        2
# 2  1|0.04675000000000002|007,2|0.0555|007 (series...        3
# 3  1|0.03425|007,2|0.03799999999999998|007 (serie...        4
# 4  1|0.04299999999999998|007,2|0.0532500000000000...        5
# 5  1|0.02975|007,2|0.02575|007 (series),3|0.02450...        6
# 6  1|0.04775000000000002|007,2|0.0494999999999999...        7
# 7  1|0.037250000000000005|007,2|0.044999999999999...        8
# 8  1|0.03949999999999998|007,2|0.0427500000000000...        9
# 9  1|0.9995|007,2|1.0|007 (series),3|0.0270000000...       10
# movie_info=
#  movieId        62423
# movie_title    62423
# genres         62423
# year           62423
# all_tag        13816
# dtype: int64  movieId cnt= 62423
# movie_info=
#     movieId                  movie_title  \
# 0        1                    Toy Story
# 1        2                      Jumanji
# 2        3             Grumpier Old Men
# 3        4            Waiting to Exhale
# 4        5  Father of the Bride Part II
# 5        6                         Heat
# 6        7                      Sabrina
# 7        8                 Tom and Huck
# 8        9                 Sudden Death
# 9       10                    GoldenEye
#
#                                         genres  year  \
# 0  Adventure|Animation|Children|Comedy|Fantasy  1995
# 1                   Adventure|Children|Fantasy  1995
# 2                               Comedy|Romance  1995
# 3                         Comedy|Drama|Romance  1995
# 4                                       Comedy  1995
# 5                        Action|Crime|Thriller  1995
# 6                               Comedy|Romance  1995
# 7                           Adventure|Children  1995
# 8                                       Action  1995
# 9                    Action|Adventure|Thriller  1995
#
#                                              all_tag
# 0  1|0.02875|007,2|0.023749999999999997|007 (seri...
# 1  1|0.041250000000000016|007,2|0.040499999999999...
# 2  1|0.04675000000000002|007,2|0.0555|007 (series...
# 3  1|0.03425|007,2|0.03799999999999998|007 (serie...
# 4  1|0.04299999999999998|007,2|0.0532500000000000...
# 5  1|0.02975|007,2|0.02575|007 (series),3|0.02450...
# 6  1|0.04775000000000002|007,2|0.0494999999999999...
# 7  1|0.037250000000000005|007,2|0.044999999999999...
# 8  1|0.03949999999999998|007,2|0.0427500000000000...
# 9  1|0.9995|007,2|1.0|007 (series),3|0.0270000000...
#
# Process finished with exit code 0

if __name__ == '__main__':
    generate_movie_data(path="./ml-25m")
    # generate_movie_data(path="./ml-1m/")
    # getUserTagMovie(path="./ml-25m")