#coding=gbk

import random
import numpy as np
import pandas as pd

import math
from operator import itemgetter
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')
train = pd.read_csv('./train_dataset.csv')
test = pd.read_csv('./test_dataset.csv')
sub = pd.read_csv('./submission.csv')

logging.info("正在执行")
data = train.copy()
data['rating'] = 1
data.head(5)
data.pivot(index='user_id', columns='item_id', values='rating')
trainSet, testSet = {}, {}
trainSet_len, testSet_len = 0, 0
pivot = 0.75

for ele in data.itertuples():
    user, item, rating = getattr(ele, 'user_id'), getattr(ele, 'item_id'), getattr(ele, 'rating')
    if random.random() < pivot:
        trainSet.setdefault(user, {})
        trainSet[user][item] = rating
        trainSet_len += 1
    else:
        testSet.setdefault(user, {})
        testSet[user][item] = rating
        testSet_len += 1
item_popular = {}
for user, items in trainSet.items():   # item:{movieID: rating}
    for item in items:
        if item not in item_popular:
            item_popular[item] = 0
        item_popular[item] += 1


item_count = len(item_popular)

# 下面建立item相似矩阵

item_sim_matrix = {}
for user, items in trainSet.items():
    for m1 in items:
        for m2 in items:
            if m1 == m2:
                continue
            item_sim_matrix.setdefault(m1, {})
            item_sim_matrix[m1].setdefault(m2, 0)
            item_sim_matrix[m1][m2] += 1     # 余弦相似度

# 计算书籍之间的相似性
for m1, related_items in item_sim_matrix.items():
    for m2, count in related_items.items():

        if item_popular[m1] == 0  or item_popular[m2] == 0:
            item_sim_matrix[m1][m2] = 0
        else:
            item_sim_matrix[m1][m2] = count / math.sqrt(item_popular[m1] * item_popular[m2])
user_lst = test['user_id'].tolist()
# 找到最相似的K本书， 最终推荐给用户
k = 208
n = 10
result = []
for user in user_lst:
    rank = {}
    watched_items = trainSet[user]  # 找出目标用户看过的书籍

    for item, rating in watched_items.items():
        for related_item, w in sorted(item_sim_matrix[item].items(), key=itemgetter(1), reverse=True)[:k]:
            # 若该物品用户看过则不推荐
            if related_item in watched_items:
                continue

            rank.setdefault(related_item, 0)
            rank[related_item] += w * float(rating)

    rec_items = sorted(rank.items(), key=itemgetter(1), reverse=True)[:n]
    for i in list(rec_items):
        result.append(i)
r = []
for i in result:
   r.append(i[0])
sub['item_id'] = r
sub
sub.to_csv('./book.csv')
print("运行终于结束啦")
