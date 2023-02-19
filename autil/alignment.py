import gc
import numpy as np
import torch
from sklearn.metrics import euclidean_distances
from scipy.spatial import distance
from torch.nn import functional

##distance##################
def mypair_distance_min(a, b, distance_type="L1"):
    ''' 求Find the distance of the entity pair, the similarity is high, the value is low'''
    if distance_type == "L1":
        return functional.pairwise_distance(a, b, p=1)  # [B*C]
    elif distance_type == "L2":
        return functional.pairwise_distance(a, b, p=2)
    elif distance_type == "L2squared":
        return torch.pow(functional.pairwise_distance(a, b, p=2), 2)
    elif distance_type == "cosine":
        return 1 - torch.cosine_similarity(a, b)  # [B*C]

def cosine_similarity3(a, b):
    '''
    a shape: [num_item_1, embedding_dim]
    b shape: [num_item_2, embedding_dim]
    return sim_matrix: [num_item_1, num_item_2]
    '''
    a = a / torch.clamp(a.norm(dim=-1, keepdim=True, p=2), min=1e-10)
    b = b / torch.clamp(b.norm(dim=-1, keepdim=True, p=2), min=1e-10)
    #sim = torch.mm(a, b.t())
    if len(a.shape) == 3: #
        sim = torch.bmm(a, torch.transpose(b, 1, 2))
    else:
        sim = torch.mm(a, b.t())
    return sim


##similarity########
def torch_sim_min_topk(embed1, embed2, top_num=1, metric='manhattan', right_size=10000):
    if embed1.is_cuda:
        left_batch_size = 1000
    else:
        left_batch_size = 5000
    links_len = embed1.shape[0]
    min_index_list = []
    for i in np.arange(0, links_len, left_batch_size):
        end = min(i + left_batch_size, links_len)
        min_index_batch = torch_sim_min_vseg(embed1[i:end, :], embed2, top_num, metric=metric, right_size=right_size)
        min_index_list.append(min_index_batch)

    min_index = torch.cat(min_index_list, 0)

    if min_index.is_cuda:
        min_index = min_index.detach().cpu().numpy()
    else:
        min_index = min_index.detach().numpy()

    return min_index


def torch_sim_min_topk_s(embed1, embed2, top_num, metric='manhattan', right_size=10000):
    if embed1.is_cuda:
        left_batch_size = 1000
    else:
        left_batch_size = 5000
    links_len = embed1.shape[0]
    min_index_list, min_scoce_list = [], []
    for i in np.arange(0, links_len, left_batch_size):
        end = min(i + left_batch_size, links_len)
        min_scoce_batch, min_index_batch = torch_sim_min_vseg(embed1[i:end, :], embed2, top_num, metric=metric,
                                                              right_size=right_size, is_score=True)
        min_scoce_list.append(min_scoce_batch)
        min_index_list.append(min_index_batch)

    min_scoce = torch.cat(min_scoce_list, 0)
    min_index = torch.cat(min_index_list, 0)

    del min_index_list, min_index_batch
    gc.collect()

    if min_index.is_cuda:
        min_scoce = min_scoce.detach().cpu().numpy()
        min_index = min_index.detach().cpu().numpy()
    else:
        min_scoce = min_scoce.detach().numpy()
        min_index = min_index.detach().numpy()

    return min_scoce, min_index

### Vertical split
def torch_sim_min_vseg(embed1, embed2, top_num, metric, right_size=10000, is_score=False):
    right_len = embed2.shape[0]
    #
    min_score_list, min_index_list = [], []
    for beg_index in np.arange(0, right_len, right_size):
        end = min(beg_index + right_size, right_len)
        min_score_batch, min_index_batch = torch_sim_min_batch(embed1, embed2[beg_index:end, :], top_num, metric=metric)
        if beg_index != 0:
            min_index_batch += beg_index
        min_score_list.append(min_score_batch)
        min_index_list.append(min_index_batch)

    min_score_merge = torch.cat(min_score_list, 1)
    min_index_merge = torch.cat(min_index_list, 1)
    #top_index = np.argsort(-min_score_merge, axis=1)  #
    top_index = min_score_merge.argsort(dim=-1, descending=False)   #
    top_index = top_index[:, :top_num]
    #
    row_count = embed1.shape[0]
    min_index = torch.zeros((row_count, top_num), )
    if is_score:
        min_score = torch.zeros((row_count, top_num))

    for i in range(row_count):
        min_index[i] = min_index_merge[i, top_index[i]]
        if is_score:
            min_score[i] = min_score_merge[i, top_index[i]]

    min_index = min_index.int()
    if is_score:
        return min_score, min_index
    else:
        return min_index


def torch_sim_min_batch(embed1, embed2, top_num, metric='manhattan'):
    ''' Find the similarity between two sets of entity lists, the more similar, the greater the value'''
    if metric == 'L1' or metric == 'manhattan':  # L1 Manhattan
        sim_mat = torch.cdist(embed1, embed2, p=1.0)  #
    elif metric == 'L2' or metric == 'euclidean':  # L2 euclidean
        sim_mat = torch.cdist(embed1, embed2, p=2.0)
    elif metric == 'cosine': #  cosine
        sim_mat = 1 - cosine_similarity3(embed1, embed2)  # [batch, net1, net1]

    if len(embed2) > top_num:
        min_score_batch, min_index_batch = sim_mat.topk(k=top_num, dim=-1, largest=False)
    else:
        min_score_batch, min_index_batch = sim_mat.topk(k=len(embed2), dim=-1, largest=False)

    del sim_mat
    gc.collect()

    return min_score_batch, min_index_batch


def task_divide(idx, n):
    ''' 划分成N个任务 '''
    total = len(idx)
    if n <= 0 or 0 == total:
        return [idx]
    if n > total:
        return [idx]
    elif n == total:
        return [[i] for i in idx]
    else:
        j = total // n
        tasks = []
        for i in range(0, (n - 1) * j, j):
            tasks.append(idx[i:i + j])
        tasks.append(idx[(n - 1) * j:])
        return tasks


def divide_batch(idx_list, batch_size):
    ''' 划分成N个任务 '''
    total = len(idx_list)
    if batch_size <= 0 or total <= batch_size:
        return [idx_list]
    else:
        n = total // batch_size
        batchs_list = []
        for i in range(n):
            beg = i * batch_size
            batchs_list.append(idx_list[beg:beg + batch_size])

        if beg + batch_size < total:
            batchs_list.append(idx_list[beg + batch_size:])
        return batchs_list


def divide_dict(idx_dict_list, batch_size):
    ''' 划分成N个任务 '''
    total = len(idx_dict_list)
    if batch_size <= 0 or total <= batch_size:
        newid_batch = np.array(idx_dict_list)
        return [newid_batch]
    else:
        n = total // batch_size
        batchs_list = []
        for i in range(n):
            beg = i * batch_size
            newid_batch = np.array(idx_dict_list[beg:beg + batch_size])
            batchs_list.append(newid_batch)

        beg = beg + batch_size
        diff = total - beg
        if diff > 0:
            if diff < batch_size/4:#
                newid_batch = batchs_list[-1]
                newid_batch = np.vstack((newid_batch, np.array(idx_dict_list[beg:])))
                batchs_list[-1] = newid_batch
            else:
                newid_batch = np.array(idx_dict_list[beg:])
                batchs_list.append(newid_batch)
        return batchs_list


def divide_array(idx_dict_array, batch_size):
    ''' 划分成N个任务 '''
    total = idx_dict_array.shape[0]
    if batch_size <= 0 or total <= batch_size:
        newid_batch = idx_dict_array
        return [newid_batch]
    else:
        n = total // batch_size
        batchs_list = []
        for i in range(n):
            beg = i * batch_size
            newid_batch = idx_dict_array[beg:beg + batch_size, :]
            batchs_list.append(newid_batch)

        beg = beg + batch_size
        diff = total - beg
        if diff > 0:
            if diff < batch_size/4:#
                newid_batch = batchs_list[-1]
                newid_batch = np.vstack((newid_batch, idx_dict_array[beg:, :]))
                batchs_list[-1] = newid_batch
            else:
                newid_batch = idx_dict_array[beg:, :]
                batchs_list.append(newid_batch)
        return batchs_list

