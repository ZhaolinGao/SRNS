import math
import tensorflow as tf
import numpy as np
import pdb
import os
import sys
from sklearn import datasets, linear_model, metrics
import random
import pickle
from time import time
from multiprocessing import Pool

def ndcg_func(ground_truths, ranks):
    result = 0
    for i, (rank, ground_truth) in enumerate(zip(ranks, ground_truths)):
        len_rank = len(rank)
        len_gt = len(ground_truth)
        idcg_len = min(len_gt, len_rank)

        # calculate idcg
        idcg = np.cumsum(1.0 / np.log2(np.arange(2, len_rank + 2)))
        idcg[idcg_len:] = idcg[idcg_len-1]

        dcg = np.cumsum([1.0/np.log2(idx+2) if item in ground_truth else 0.0 for idx, item in enumerate(rank)])
        result += dcg / idcg
    return result / len(ranks)

def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(actual)
    true_users = 0
    for i, v in actual.items():
        act_set = set(v)
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    assert num_users == true_users
    return sum_recall / true_users

def eval(model, sess, train_data, test_data, num_user, num_item):
    batch_size = 5
    current_user = 0
    predictions = []

    for i in range(num_user//batch_size + 1):
        if current_user+batch_size >= num_user:
            users = np.arange(current_user, num_user)
            size = num_user-current_user
        else:
            users = np.arange(current_user, current_user+batch_size)
            current_user += batch_size
            size = batch_size

        users = np.expand_dims(np.repeat(users, num_item), axis=1)
        items = np.expand_dims(np.arange(num_item), axis=1)
        items = np.repeat(items, size, axis=1).transpose().reshape((-1, 1))

        feed_dict = {model.user_input:users,
                     model.item_input_pos:items}
        predictions.append(np.reshape(sess.run([model.score],feed_dict),[size, num_item]))

    predictions = np.concatenate(predictions, axis=0)

    topk = 20
    predictions[train_data.nonzero()] = np.NINF

    ind = np.argpartition(predictions, -topk)
    ind = ind[:, -topk:]
    arr_ind = predictions[np.arange(len(predictions))[:, None], ind]
    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(predictions)), ::-1]
    pred_list = ind[np.arange(len(predictions))[:, None], arr_ind_argsort]

    recall = []
    for k in [5, 10, 20]:
        recall.append(recall_at_k(test_data, pred_list, k))

    all_ndcg = ndcg_func([*test_data.values()], pred_list[list(test_data.keys())])
    ndcg = [all_ndcg[x-1] for x in [5, 10, 20]]

    return recall, ndcg

# def eval_one_user(id):
#     score = _score_eval[id, 1:]
#     target_score = _score_eval[id, 0]
#     rank = 0
#     for each in score:
#         if each>target_score:
#             rank+=1
#         if rank>3:
#             break
#     recall = []
#     ndcg = []
#     for topk in [1,3]:
#         if rank<topk:
#             recall.append(1)
#             ndcg.append(math.log(2)/math.log(rank+2))
#         else:
#             recall.append(0)
#             ndcg.append(0)
#     return recall, ndcg


# def eval(model, sess, test_data, test_data_neg):
#     global _score_eval
#     global _test_data
#     global _test_data_neg
#     _test_data = test_data
#     _test_data_neg = test_data_neg
#     score_eval = []    

#     for i in range(test_data.shape[0]):
#         user_input = np.ones((101,1))*test_data[i,0]
#         item_input = np.reshape(np.array([test_data[i,1]]+test_data_neg[i].tolist()),[-1,1])
#         feed_dict = {model.user_input:user_input,
#                      model.item_input_pos:item_input}
#         score_eval.append(np.reshape(sess.run([model.score],feed_dict),[1,-1]))
#     score_eval = np.concatenate(score_eval, axis=0)
#     _score_eval = score_eval

#     pool = Pool(20)
#     res = pool.map(eval_one_user, range(test_data.shape[0]))
#     pool.close()
#     pool.join()

#     Recall = []
#     NDCG = []
#     for i in range(2): 
#         Recall.append(np.mean(np.array([r[0][i] for r in res])))
#         NDCG.append(np.mean(np.array([r[1][i] for r in res])))
        
#     return Recall, NDCG


def predict_fast(model, sess, num_user, num_item, parallel_users, predict_data=None):
    scores = []
    for s in range(int(num_user/parallel_users)):
        user_input = []
        item_input = []
        for i in range(s*parallel_users,(s+1)*parallel_users):
            user_input.append(np.ones((predict_data.shape[1], 1)) * i)
            item_input.append(np.reshape(predict_data[i], [-1, 1]))
        user_input = np.concatenate(user_input,axis=0)
        item_input = np.concatenate(item_input, axis=0)
        feed_dict = {model.user_input: user_input,
                     model.item_input_pos: item_input}
        scores.append(np.reshape(sess.run([model.score], feed_dict), [parallel_users, -1]))
    if int(num_user / parallel_users) * parallel_users < num_user:
        user_input = []
        item_input = []
        for i in range(int(num_user/parallel_users)*parallel_users,num_user):
            user_input.append(np.ones((predict_data.shape[1], 1)) * i)
            item_input.append(np.reshape(predict_data[i], [-1, 1]))
        user_input = np.concatenate(user_input, axis=0)
        item_input = np.concatenate(item_input, axis=0)
        feed_dict = {model.user_input: user_input,
                     model.item_input_pos: item_input}
        scores.append(np.reshape(sess.run([model.score], feed_dict),
                                 [num_user-int(num_user/parallel_users)*parallel_users, -1]))
    scores = np.concatenate(scores, axis=0)

    return scores

def predict_pos(model, sess, num_user, max_posid, parallel_users, predict_data=None):
    scores = []
    for s in range(int(num_user/parallel_users)):
        user_input = []
        item_input = []
        for i in range(s*parallel_users,(s+1)*parallel_users):
            user_input.append(np.ones((len(predict_data[i]), 1)) * i)
            item_input.append(np.reshape(predict_data[i], [-1, 1]))
        user_input = np.concatenate(user_input,axis=0)
        item_input = np.concatenate(item_input, axis=0)
        feed_dict = {model.user_input: user_input,
                     model.item_input_pos: item_input}
        score_flatten = sess.run(model.score, feed_dict)
        score_tmp = np.zeros((parallel_users, max_posid))

        c = 0
        for i in range(s * parallel_users, (s + 1) * parallel_users):
            l = len(predict_data[i])
            score_tmp[i-s*parallel_users,0:l] = \
                np.reshape(score_flatten[c:c+l, 0], [1, -1])
            c += l
        scores.append(score_tmp)

    if int(num_user / parallel_users) * parallel_users < num_user:
        user_input = []
        item_input = []
        for i in range(int(num_user / parallel_users) * parallel_users, num_user):
            user_input.append(np.ones((len(predict_data[i]), 1)) * i)
            item_input.append(np.reshape(predict_data[i], [-1, 1]))
        user_input = np.concatenate(user_input, axis=0)
        item_input = np.concatenate(item_input, axis=0)
        feed_dict = {model.user_input: user_input,
                     model.item_input_pos: item_input}
        score_flatten = sess.run(model.score, feed_dict)
        score_tmp = np.zeros((num_user - int(num_user / parallel_users) * parallel_users, max_posid))

        c = 0
        for i in range(int(num_user / parallel_users) * parallel_users, num_user):
            l = len(predict_data[i])
            score_tmp[i - int(num_user / parallel_users) * parallel_users, 0:l] = \
                np.reshape(score_flatten[c:c + l, 0], [1, -1])
            c += l
        scores.append(score_tmp)
    scores = np.concatenate(scores, axis=0)

    return scores





