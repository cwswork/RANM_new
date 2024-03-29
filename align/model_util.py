import copy
import gc
import math
import torch
import numpy as np

from autil import alignment


def links_init_partiton(train_links, ent_new_pair):
    train_links_new = []
    for e1, e2 in train_links:
        if (e1, e2) in ent_new_pair.keys():
            new_pair = ent_new_pair[(e1, e2)]  # new_id
        else:
            new_pair = (e1, e2)
        train_links_new.append(new_pair)
    return train_links_new


def gen_link_neg(tt_links_new, tt_candidates, tt_candidates_right, neg_k, is_cuda):
    neg_pair = []
    for pe1, pe2 in tt_links_new:
        neg_indexs1 = copy.deepcopy(tt_candidates[pe1])  #
        if pe2 in neg_indexs1:
            neg_indexs1.remove(pe2)
        neg_indexs2 = copy.deepcopy(tt_candidates_right[pe2])  #
        if pe1 in neg_indexs2:
            neg_indexs2.remove(pe1)

        for i in range(neg_k):
            neg_pair.append((pe1, pe2, neg_indexs1[i], neg_indexs2[i]))  # (pe1,pe2) is aligned entity pair, (pe1,ne2) is negative sample

    neg_pair = torch.LongTensor(neg_pair)  # eer_adj_data
    if is_cuda:
        neg_pair = neg_pair.cuda()

    return neg_pair


def gen_neg(ent_embed, tt_links_new, metric, neg_k):
    es1 = [e1 for e1,e2 in tt_links_new]
    es2 = [e2 for e1,e2 in tt_links_new] # ???e1

    neg1_array = gen_neg_each(ent_embed, es1, metric, neg_k)
    neg2_array = gen_neg_each(ent_embed, es2, metric, neg_k)

    neg_pair = []
    for i in range(len(es2)):
        e1, e2 = tt_links_new[i]
        for j in range(neg_k):
            neg_pair.append((e1, e2, neg1_array[i][j], neg2_array[i][j]))

    neg_pair = torch.LongTensor(np.array(neg_pair))  # eer_adj_data
    if ent_embed.is_cuda:
        neg_pair = neg_pair.cuda()

    return neg_pair

def gen_neg_noName(tt_links_new, neg_k):
    es1_list = [e1 for e1, e2 in tt_links_new]
    es2_list = [e2 for e1, e2 in tt_links_new]
    #all_ent = list(range(kg_E))
    neg_pair = []
    for i in range(len(tt_links_new)):
        e1, e2 = tt_links_new[i]
        neg1 = np.random.choice(es1_list, neg_k+1).tolist()
        if e1 in neg1:
            neg1.remove(e1)

        neg2 = np.random.choice(es2_list, neg_k + 1).tolist()
        if e2 in neg2:
            neg2.remove(e2)

        for j in range(neg_k):
            neg_pair.append((e1, e2, neg1[j], neg2[j]))

    neg_pair = torch.LongTensor(np.array(neg_pair))  # eer_adj_data
    return neg_pair


def gen_neg_each(ent_embed, left_ents, metric, neg_k):
    min_index = alignment.torch_sim_min_topk(ent_embed[left_ents, :], ent_embed, top_num=neg_k + 1, metric=metric)

    e_t = len(left_ents)
    neg = []
    for i in range(e_t):
        rank = min_index[i, :].tolist()
        if left_ents[i] == rank[0]:
            rank = rank[1:]
        else:
            if left_ents[i] in rank:
                rank.remove(left_ents[i])
            else:
                rank = rank[:neg_k]
        neg.append(rank)

    neg = np.array(neg) # neg.reshape((e_t * self.neg_k,))
    return neg  # (n*k,)



def link_inset(list1):
    list2 = []
    for i in list1:
        if i not in list2:
            list2.append(i)
    return list2

