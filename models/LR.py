import pandas as pd
import numpy as np
from datetime import datetime
from time import time
import math

def generate_train_data():
    # 处理训练数据
    df = pd.read_csv("../data/ml-25m/ratings.csv", sep=',', nrows=10000,
                     names=["userId", "movieId", "rating", "timestamp"], skiprows=1)
    tags = pd.read_csv("../data/ml-25m/")