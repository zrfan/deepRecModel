import pandas as pd
import numpy as np
from datetime import datetime
from time import time
import math

def generate_train_data():
    # 处理训练数据：dict{user_id:: {item_id:rating}}
    df = pd.read_csv("../data/ml-25m/ratings.csv", sep=',', nrows=100000,
                     names=["userId", "movieId", "rating", "timestamp"], skiprows=1)
    # d为每个用户看过的电影及其打分的列表
    d = dict()
    for _, row in df.iterrows():
        user_id = str(int(row['userId']))
        item_id = str(int(row['movieId']))
        rating = str(row['rating'])
        if user_id not in d.keys():
            d[user_id] = {item_id: rating}
        else:
            d[user_id][item_id] = rating
    print("user len=", len(d))
    # print(d.keys())
    return d

def process_data(train_data):
    N = dict()  # N{item: userNum} ,统计每个item的观看用户数
    C = dict()  # C{item: {item:simUserNum}}   item与item的共同用户数
    for u, items in train_data.items():
        for i in items.keys():
            if N.get(i, -1) == -1:  # 对user 的每个item，N[i]只有一个
                N[i] = 0
            N[i] += 1
            if C.get(i, -1) == -1:
                C[i] = dict()
            for j in items.keys():
                if i == j:
                    continue
                if C[i].get(j, -1) == -1:
                    C[i][j] = 0
                C[i][j] += 1
    return N, C

def item_sim(C, N): # 计算相似度矩阵，杰卡德算法, 最终结果保存到C中 {item: {item: sim_score}}
    for i, relate_items in C.items():
        for j, c_ij in relate_items.items():
            C[i][j] = 2.0 * c_ij / (N[i]+N[j]+1.0)
    return C



def recommendation(train_data, user_id, C, k):
    rank = dict()  # rank {item: score} 存储item候选集, 指待推荐给指定user的items
    # 用户user_id有很多已经观看过的item，每个item取k个相似item
    Ru = train_data[user_id]
    for i, rating in Ru.items():  # 相比于user_based, item_based的一个用户对应多个item，所以大循环要遍历item
        for j, sim_score in sorted(C[i].items(), key=lambda x: x[1], reverse=True)[0:k]:
            # 相对于user_based，对每个user还要遍历各自的item，item_based则不需要，因为相似度矩阵就是item-item
            # 过滤这个user已经评价过的item
            if j in Ru:
                continue
            if rank.get(j, -1) == -1:
                rank[j] = 0
            rank[j] += sim_score*float(rating)
    return  rank

if __name__ == '__main__':
    train_data = generate_train_data()
    N, C = process_data(train_data)
    C = item_sim(C, N)

    user_id = '3'
    k = 5
    rank = recommendation(train_data, user_id, C, k)
    for i, score in sorted(rank.items(), key=lambda p: p[1], reverse=True)[0:20]:
        print("item=", i, " score=", score)








