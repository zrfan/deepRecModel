import pandas as pd
import numpy as np
from datetime import datetime
import time
import math
import csv
from data_process import getYear

def generate_user_features(path):
    user_info = pd.read_csv(path+"/users.dat", header=None, encoding='utf-8', delimiter="::", quoting=csv.QUOTE_NONE,
                            names=["userId", "gender", "age", "occupation", "zipcode"])

    print(user_info.head(10))
def generate_movie_feature(path):
    movie_info = pd.read_csv(path + "/movies.dat", header=None, delimiter="::", quoting=csv.QUOTE_NONE,
                             names=["movieId", "movie_title", "genres"])

    movie_info["year"] = movie_info["movie_title"].apply(lambda x: getYear(x))
    print(movie_info.head(50))
# occupationDict = {*  0:  "other" or not specified
# 	*  1:  "academic/educator"
# 	*  2:  "artist"
# 	*  3:  "clerical/admin"
# 	*  4:  "college/grad student"
# 	*  5:  "customer service"
# 	*  6:  "doctor/health care"
# 	*  7:  "executive/managerial"
# 	*  8:  "farmer"
# 	*  9:  "homemaker"
# 	* 10:  "K-12 student"
# 	* 11:  "lawyer"
# 	* 12:  "programmer"
# 	* 13:  "retired"
# 	* 14:  "sales/marketing"
# 	* 15:  "scientist"
# 	* 16:  "self-employed"
# 	* 17:  "technician/engineer"
# 	* 18:  "tradesman/craftsman"
# 	* 19:  "unemployed"
# 	* 20:  "writer"}
if __name__ == '__main__':
    generate_movie_feature(path="./ml-1m/")