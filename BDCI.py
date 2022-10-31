#coding=gbk
import random
import pandas as pd
import math
from operator import itemgetter
train = pd.read_csv('./train_dataset.csv')
test = pd.read_csv('./test_dataset.csv')
yangli = pd.read_csv('./submission.csv')
data = train.copy()
data['score'] = 1
data.head(5)
data.pivot(index='user_id', columns='item_id', values='score')
xunliangji, ceshiji = {}, {}
xunliangji_len, ceshiji_len = 0, 0
pivot = 0.8


#上面导包和读取数据集，再切分数据集

for ys in data.itertuples():
    user, item, score = getattr(ys, 'user_id'), getattr(ys, 'item_id'), getattr(ys, 'score')
    if random.random() < pivot:
        xunliangji.setdefault(user, {})
        xunliangji[user][item] = score
        xunliangji_len += 1
    else:
        ceshiji.setdefault(user, {})
        ceshiji[user][item] = score
        ceshiji_len += 1
bookscore = {}
#计算相似度
for user, items in xunliangji.items():
    for item in items:
        if item not in bookscore:
            bookscore[item] = 0
        bookscore[item] += 1

xsi = {}
for user, items in xunliangji.items():
    for book1 in items:
        for book2 in items:
            if book1 == book2:
                continue
            xsi.setdefault(book1, {})
            xsi[book1].setdefault(book2, 0)
            xsi[book1][book2] += 1     # 余弦相似度

# 计算书籍之间的相似性
for book1, xguang in xsi.items():
    for book2, count in xguang.items():

        if bookscore[book1] == 0  or bookscore[book2] == 0:
            xsi[book1][book2] = 0
        else:
            xsi[book1][book2] = count / math.sqrt(bookscore[book1] * bookscore[book2])
user_lst = test['user_id'].tolist()
m = 208
h = 10
result = []
for user in user_lst:
    rank = {}
    watched_items = xunliangji[user]

    for item, score in watched_items.items():
        for xg, w in sorted(xsi[item].items(), key=itemgetter(1), reverse=True)[:m]:

            if xg in watched_items:
                continue

            rank.setdefault(xg, 0)
            rank[xg] += w * float(score)

    rec_items = sorted(rank.items(), key=itemgetter(1), reverse=True)[:h]
    for p in list(rec_items):
        result.append(p)
n = []
for p in result:
   n.append(p[0])
#生成推荐和提交文件
yangli['item_id'] = n
yangli.to_csv('./book.csv')
print("运行终于结束啦")
