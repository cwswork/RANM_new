import gc
import numpy as np
import torch
from torch.nn import functional

##distance##################
def mypair_distance_min(a, b, distance_type="L1"):
    ''' Find the distance of the entity pair, the similarity is high, the value is low '''
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
    if len(a.shape) == 3: # The matrix is three-dimensional
        sim = torch.bmm(a, torch.transpose(b, 1, 2))
    else:
        sim = torch.mm(a, b.t())
    return sim


##similarity########
def torch_sim_max_topk(embed1, embed2, top_num, metric='manhattan'):
    if embed1.is_cuda:
        left_batch_size = 1000
    else:
        left_batch_size = 5000
    links_len = embed1.shape[0]
    max_index_list = []
    for i in np.arange(0, links_len, left_batch_size):
        end = min(i + left_batch_size, links_len)
        max_index_batch = torch_sim_max_vseg(embed1[i:end, :], embed2, top_num, metric=metric)
        max_index_list.append(max_index_batch)

    max_index = torch.cat(max_index_list, 0)
    # max_index = np.concatenate(max_index_list, axis=0)

    del max_index_list, max_index_batch
    gc.collect()  # Garbage collection

    if max_index.is_cuda:
        max_index = max_index.detach().cpu().numpy()
    else:
        max_index = max_index.detach().numpy()

    return max_index


def torch_sim_max_topk_s(embed1, embed2, top_num, metric='manhattan'):
    if embed1.is_cuda:
        left_batch_size = 1000
    else:
        left_batch_size = 5000
    links_len = embed1.shape[0]
    max_index_list, max_scoce_list = [], []
    for i in np.arange(0, links_len, left_batch_size):
        end = min(i + left_batch_size, links_len)
        max_index_batch, max_scoce_batch = torch_sim_max_vseg(embed1[i:end, :], embed2, top_num, metric=metric, is_scoce=True)
        max_index_list.append(max_index_batch)
        max_scoce_list.append(max_scoce_batch)

    max_index = torch.cat(max_index_list, 0)
    max_scoce = torch.cat(max_scoce_list, 0)
    # max_index = np.concatenate(max_index_list, axis=0)

    del max_index_list, max_index_batch, max_scoce_list, max_index_batch
    gc.collect()  # Garbage collection

    if max_index.is_cuda:
        max_index = max_index.detach().cpu().numpy()
        max_scoce = max_scoce.detach().cpu().numpy()
    else:
        max_index = max_index.detach().numpy()
        max_scoce = max_scoce.detach().numpy()

    return max_index, max_scoce


### Vertical split
def torch_sim_max_vseg(embed1, embed2, top_num, metric, is_scoce=False):
    right_len = embed2.shape[0]
    batch_size = 10000 # int(right_len/right_num)

    #max_index_merge, max_scoce_merge = None, None
    max_index_list, max_scoce_list = [], []
    for beg_index in np.arange(0, right_len, batch_size):
        end = min(beg_index + batch_size, right_len)
        max_scoce_batch, max_index_batch = torch_sim_max_batch(embed1, embed2[beg_index:end, :], top_num, metric=metric)
        if beg_index != 0:
            max_index_batch += beg_index
        max_index_list.append(max_index_batch)
        max_scoce_list.append(max_scoce_batch)

    max_scoce_merge = torch.cat(max_scoce_list, 1)
    max_index_merge = torch.cat(max_index_list, 1)
    # Combine, take top_num
    top_index = max_scoce_merge.argsort(dim=-1, descending=True)
    top_index = top_index[:, :top_num]
    #
    row_count = embed1.shape[0]
    max_index = torch.zeros((row_count, top_num), )
    if is_scoce:
        max_scoce = torch.zeros((row_count, top_num))

    for i in range(row_count):
        max_index[i] = max_index_merge[i, top_index[i]]
        if is_scoce:
            max_scoce[i] = max_scoce_merge[i, top_index[i]]

    max_index = max_index.int()
    if is_scoce:
        return max_index, max_scoce
    else:
        return max_index


def torch_sim_max_batch(embed1, embed2, top_num, metric='manhattan'):
    ''' Find the similarity between two sets of entity lists, the more similar, the greater the value '''
    if metric == 'L1' or metric == 'manhattan':  # L1 Manhattan
        sim_mat = - torch.cdist(embed1, embed2, p=1.0)  # 1-
    elif metric == 'L2' or metric == 'euclidean':  # L2 euclidean
        sim_mat = - torch.cdist(embed1, embed2, p=2.0)
    elif metric == 'cosine': # Cosine similarity
        sim_mat = cosine_similarity3(embed1, embed2)  # [batch, net1, net1]

    if len(embed2) > top_num:
        max_scoce_batch, max_index_batch = sim_mat.topk(k=top_num, dim=-1, largest=True)  # top_num
    else:
        max_scoce_batch, max_index_batch = sim_mat.topk(k=len(embed2), dim=-1, largest=True)  # top_num

    del sim_mat
    gc.collect()
    return max_scoce_batch, max_index_batch


def task_divide(idx, n):
    ''' Divide into N tasks '''
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


def divide_array(idx_dict_array, batch_size):
    ''' Divide into N tasks '''
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

