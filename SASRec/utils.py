import sys
import copy
import torch
import random,math
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue
import pandas as pd
sys.path.append("..")
from prompts import Prompts
from chatagent import *
from loguru import logger
logger.add("./utils.log", backtrace=True, diagnose=True)

# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample():

        user = np.random.randint(1, usernum + 1)
        while  True: 
            try:
                user = np.random.randint(1, usernum + 1)
                if len(user_train[user]) > 1:
                    break
            except:
                user = np.random.randint(1, usernum + 1)
                #print()

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        return (user, seq, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


# train/val/test data generation
def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}

    user_set=pd.read_csv('data/douban/moviedata/'+'users.csv', encoding='utf-8',on_bad_lines='skip')
    user_set=user_set.astype(str)
    train_set = pd.read_csv('data/douban/moviedata/'+fname+'.csv', encoding='utf-8',on_bad_lines='skip')
    item_set = pd.read_csv('data/douban/moviedata/'+'movie_set.csv', encoding='utf-8',on_bad_lines='skip')
    #test_set = pd.read_csv('data/'+'test_set.csv', encoding='utf-8',on_bad_lines='skip')
    md5_id_dict =user_set.set_index('user_md5')['id'].to_dict()
    for index, row in train_set.iterrows():
        md5= row['user_md5']
        id = int(md5_id_dict[md5])
        like_str = row['like']
        if like_str=='nan' or str(like_str).count('|')<3:
            continue
        User[id] =[int(x) for x in like_str.split("|")]  
        usernum=max(id,usernum)
        itemnum=max(itemnum,max(User[id]))

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum,User,item_set]
class Metric:
    NDCG=0.0
    HT=0.0
# TODO: merge evaluate functions for test and val set
# evaluate on test set
def evaluate(model, dataset, args):
    [train, valid, test, usernum, itemnum,user_dic,item_set] = copy.deepcopy(dataset)
    mtc =[Metric(),Metric(),Metric()]
    valid_user = 0.0
    

    if len(user_dic)>10000:
        users = random.sample(user_dic.keys(), 10000)
    else:
        users = user_dic.keys()
    for u in users:
        rank=[99,99,99]
        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)#用户历史喜好
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]] #候选项
        for _ in range(50):
            t = random.choice(item_set['movie_id'])#随机加入不在训练集中的对象
            while t in rated: t = random.choice(item_set['movie_id'])
            item_idx.append(t)
        #print('目标',test[u][0])  
        #print('历史',seq)
        #print('候选',item_idx)
        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        
        predictions = predictions[0] # - for 1st argsort DESC
        #print('predic',predictions)
        #print('***')
        rank[0] = predictions.argsort().argsort()[0].item()
        #print('rank[0]',rank[0])
        valid_user += 1
        
        
        predictions_cpu = predictions.cpu().detach().numpy()
        top_five_indices = predictions_cpu.argsort()[-5:] #前五个的序号
        
        top_five_items = [item_idx[i] for i in top_five_indices]#前五个的movie_id
        #print('前五个的movie_id',top_five_items)
        #print(item_idx)
        #print('sasrec 前五id',top_five_items)
        #llm rerank
        random.shuffle(item_idx)
        result = rerank(item_idx,seq,[],[])
        #print('qwen 前五id',result)
        for k in range(len(result)):
            if result[k] == test[u][0]:
                rank[1]=k
        #llm与SASREC混合rerank      
        result2 = rerank(item_idx,seq,[],top_five_items)
        #print('qwen+sasrec 前五id',result2)
        for k in range(len(result2)):
            if result2[k] == test[u][0]:
                rank[2]=k
        
        print(valid_user)
        for j in range(3):
            print('rank:',j,':',rank[j])
            logger.info(f"rank{str(j)}:{str(rank[j])}")
            if rank[j] < 5:
                mtc[j].NDCG += 1 / np.log2(rank[j] + 2)
                mtc[j].HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()
        logger.info(f"round:{str(valid_user)}")
        for item in mtc:
            ndcg=item.NDCG/valid_user
            ht=item.HT/valid_user
            print("ndcg:",ndcg)
            print("ht:",ht)
            logger.info(f"ndcg:{str(ndcg)} , ht:{str(ht)}")

    for item in mtc:
        item.NDCG/=valid_user
        item.HT/=valid_user
    
    return mtc


# evaluate on val set
def evaluate_valid(model, dataset, args):
    [train, valid, test, usernum, itemnum,user_dic,item_set] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if len(user_dic)>10000:
        users = random.sample(user_dic.keys(), 10000)
    else:
        users = user_dic.keys()
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user