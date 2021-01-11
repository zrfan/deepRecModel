import pandas as pd
import numpy as np
from datetime import datetime
from time import time
import math

def mergeTag(df, column="all_tag"):
    tags = list(df[column])
    return ','.join(tags)

def generate_train_data():
    # 处理训练数据
    user_rating_movie = pd.read_csv("../data/ml-25m/ratings.csv", sep=',', nrows=10000, names=["userId", "movieId", "rating", "timestamp"], skiprows=1)
    user_tag_movie = pd.read_csv("../data/ml-25m/tags.csv", sep=',', nrows=100000, names=["userId", "movieId", "tag", "timestamp"], skiprows=1)
    movie_tag_score = pd.read_csv("../data/ml-25m/genome-scores.csv", names=["movieId", "tagId", "tag_relevance"], skiprows=1)
        # .sort_values(by=['movieId', 'tag_relevance'], ascending=[True, False])
    tag_info = pd.read_csv("../data/ml-25m/genome-tags.csv", names=["tagId", "tag_name"], skiprows=1)
    movie_info = pd.read_csv("../data/ml-25m/movies.csv", names=["movieId", "movie_title", "genres"], skiprows=1)

    print("movie_tag size=", movie_tag_score.size, "tag_info size=", tag_info.size, "\n", "movie_info size=", movie_info.size)

    tag_info = pd.merge(movie_tag_score, tag_info, how='left', on=['tagId'])  # [movieId, tagId, tag_relevance, tag_name]
    print("movie_tag_info before group size=", tag_info.size)
    tag_info["all_tag"] = tag_info['tagId'].map(str) + '|' + tag_info["tag_relevance"].map(str) + "|" + tag_info["tag_name"]
    tag_info = tag_info[["movieId", "all_tag"]]
    print("movie_tag_info before group size=", tag_info.size, "\n", tag_info.head(10))
    movie_tag_info = tag_info.groupby('movieId').apply(mergeTag, column="all_tag")
    movie_tag_info = pd.DataFrame({'movieId': movie_tag_info.index, "all_tag": movie_tag_info.values})

    print("movie_tag_info=", movie_tag_info.size, "\n", movie_tag_info.head(10))

    movie_info = pd.merge(movie_info, movie_tag_info, how='left', on=['movieId'])   # [movieId, movie_title, genres, all_tag]
    print("movie_info=", movie_info.size, "\n", movie_info.head(10))

# movie_tag size= 46753344 tag_info size= 2256
# movie_info size= 187269
# movie_tag_info before group size= 62337792
# movie_tag_info before group size= 31168896

if __name__ == '__main__':
    generate_train_data()