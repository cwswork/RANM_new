import copy
import torch
import numpy as np
from autil import alignment

## Round1
# 6 Generate candidate set——ent_candidates
def gen_candidate(ename_embed, train_links_array, valid_links_array, test_links_array, candidate_num,
                         metric='L1', is_cuda=True):
    if isinstance(train_links_array, list):
        train_links_array = np.array(train_links_array)
    if isinstance(test_links_array, list):
        test_links_array = np.array(test_links_array)

    train_candidates = candidate_partition(ename_embed, train_links_array[:, 0], train_links_array[:, 1],
                                      candidate_num, metric=metric, is_cuda=is_cuda)  # 3000*list(100)
    test_candidates = candidate_partition(ename_embed, test_links_array[:, 0], test_links_array[:, 1],
                                      candidate_num, metric=metric, is_cuda=is_cuda)

    test_candidates.update(train_candidates)
    if valid_links_array != []:
        if isinstance(valid_links_array, list):
            valid_links_array = np.array(valid_links_array)
        valid_candidates = candidate_partition(ename_embed, valid_links_array[:, 0], valid_links_array[:, 1],
                                          candidate_num, metric=metric, is_cuda=is_cuda)
        test_candidates.update(valid_candidates)

    return test_candidates

def gen_candidate_double(ent_emb, ents1, ents2, candidate_num, metric='L1', is_cuda=True):

    emb1 = ent_emb[ents1]
    emb2 = ent_emb[ents2]
    if emb2.is_cuda == False and is_cuda:
        emb1 = emb1.cuda()
        emb2 = emb2.cuda()

    # kg1
    min_index = alignment.torch_sim_min_topk(emb1, emb2, top_num=candidate_num, metric=metric)
    ent2candidates = dict()  # {e1: e2_list}
    ents2array2 = np.array(ents2)
    for i in range(len(min_index)):
        e1 = ents1[i]
        e2_list = ents2array2[min_index[i]].tolist()
        ent2candidates[e1] = e2_list
    # kg2
    min_index = alignment.torch_sim_min_topk(emb2, emb1, top_num=candidate_num, metric=metric)
    ents2array1 = np.array(ents1)
    for i in range(len(min_index)):
        e2 = ents2[i]
        e1_list = ents2array1[min_index[i]].tolist()
        ent2candidates[e2] = e1_list

    return ent2candidates


def candidate_partition(ent_emb, ents1, ents2, candidate_num_1, metric='L1', is_cuda=True):
    """
    return a dict, key = entity, value = candidates (likely to be aligned entities) {e1: e2_list}
    """
    emb1 = ent_emb[ents1]
    emb2 = ent_emb[ents2]
    if emb2.is_cuda == False and is_cuda:
        emb1 = emb1.cuda()
        emb2 = emb2.cuda()

    min_index = alignment.torch_sim_min_topk(emb1, emb2, top_num=candidate_num_1, metric=metric)
    ent2candidates = dict()  # {e1: e2_list}
    ents2array = np.array(ents2)
    for i in range(len(min_index)):
        e1 = ents1[i]
        e2_list = ents2array[min_index[i]].tolist()
        ent2candidates[e1] = e2_list

    return ent2candidates


def batch_topk(mat, candidate_num=50, largest=True, is_cuda=True, bs=2048):
    res_index = []
    axis_0 = mat.shape[0]  # N
    for i in range(0, axis_0, bs):
        if is_cuda:
            temp_div_mat = mat[i:min(i + bs, axis_0)].cuda()
        else:
            temp_div_mat = mat[i:min(i + bs, axis_0)]

        score_mat, index_mat = temp_div_mat.topk(candidate_num, largest=largest)
        res_index.append(index_mat.cpu())
        #res_score.append(score_mat_new.cpu())
    res_index = torch.cat(res_index, 0)
    #res_score = torch.cat(res_score, 0)
    return res_index # , res_score


## Round2 According to the new temporary neighbor relationship, obtain a new candidate set

# 7 Obtain matching neighbors of candidate entity pairs
def get_match_rel(ent_neigh_dict, ent_candidates, kg1_ent, kg2_ent, myconfig):
    ent_rset = dict()  # e1，e2 set

    # KG1
    for e1 in kg1_ent:
        ent_rset[e1] = set([r for e, r in ent_neigh_dict[e1]])
    ent_rset1_len = len(ent_rset)
    myconfig.myprint('ent_rset1_len: ' + str(ent_rset1_len))
    # KG2
    for e2 in kg2_ent:
        ent_rset[e2] = set([r for e, r in ent_neigh_dict[e2]])
    ent_rset2_len = len(ent_rset) - ent_rset1_len
    myconfig.myprint('ent_rset2_len: ' + str(ent_rset2_len))

    #  ent_candidates：dict{E:list(100)}
    kg1_ents = np.array(list(ent_candidates.keys()))  # E
    kg2_candidates_array = np.array(list(ent_candidates.values()))  # (E, 100)
    #
    nums_threads = 4
    if nums_threads > 1:
        rests = list()
        search_tasks = alignment.task_divide(range(kg1_ents.shape[0]), nums_threads)  # 15000
        pool = torch.multiprocessing.Pool(processes=len(search_tasks))
        for rowids in search_tasks:
            rests.append(pool.apply_async(get_match_rel_batch, (kg1_ents[rowids], kg2_candidates_array[rowids],
                                                                ent_rset, ent_neigh_dict)))
        pool.close()  #
        pool.join()  #

        rel_match_dict = dict()
        for rest in rests:
            batch_match_dict = rest.get()
            rel_match_dict.update(batch_match_dict)
    else:
        rel_match_dict = get_match_rel_batch(kg1_ents, kg2_candidates_array, ent_rset, ent_neigh_dict)

    myconfig.myprint('rel_match_dict: ' + str(len(rel_match_dict)))
    myconfig.myprint('ent_rset: ' + str(len(ent_rset)))

    return rel_match_dict, ent_rset  # [r_union, e1_union_list, e2_union_list]

# multi-process, Obtain matching neighbors of candidate entity pairs
def get_match_rel_batch(kg1_ents, kg2_candidates_array, ent_rset, ent_neigh_dict):
    match_dict = dict()
    for e1_row in range(len(kg1_ents)):
        e1 = kg1_ents[e1_row]
        rset1 = ent_rset[e1]
        for e2 in kg2_candidates_array[e1_row]:
            r_union = rset1 & ent_rset[e2]
            if len(r_union) > 0:
                e1_union_list = [(e1, r1) for (e1, r1) in ent_neigh_dict[e1] if r1 in r_union]
                e2_union_list = [(e2, r2) for (e2, r2) in ent_neigh_dict[e2] if r2 in r_union]
                match_dict[(e1, e2)] = [r_union, e1_union_list, e2_union_list]

    return match_dict


# 8 Get a new temporary neighbor relationship —— newentid_dict
def get_candidate_new(ent_candidates, ent_neigh_dict, temp_rel_match_dict, oldent_rset,
                      kg_E, kg1_ent, kg2_ent):
    # All entities (new and old) correspond to the oldID number
    newentid_dict = {}
    for oldid in range(kg_E):
        newentid_dict[oldid] = oldid

    # The new ID list corresponding to the old ID of all entities, used to add the new entity to the candidate list
    temp_ent_old2list_dict = dict()
    for oldid in range(kg_E):
        temp_ent_old2list_dict[oldid] = [oldid]

    # The associated R set of all new ID entities, each original is a set of id correspondences
    newent_reset = dict()
    for e_old, rset in oldent_rset.items():
        newent_reset[e_old] = dict()

    temp_pairs_union_dict = dict()  # Matching neighbor
    ent_new_pair = dict()
    newnet_id = kg_E
    kg1_ent_new, kg2_ent_new = copy.deepcopy(kg1_ent), copy.deepcopy(kg2_ent)
    for e1_old, e2_list in ent_candidates.items():
        for e2_old in e2_list:
            e1_new, e2_new = e1_old, e2_old
            if (e1_old, e2_old) in temp_rel_match_dict.keys():
                [r_union, e1_union_list, e2_union_list] = temp_rel_match_dict[(e1_old, e2_old)]
                if r_union != oldent_rset[e1_old]:
                    if r_union not in newent_reset[e1_old].values():
                        e1_new = newnet_id
                        newentid_dict[e1_new] = e1_old
                        ent_neigh_dict[e1_new] = e1_union_list
                        newent_reset[e1_old][e1_new] = r_union
                        temp_ent_old2list_dict[e1_old].append(e1_new)
                        kg1_ent_new.append(newnet_id)  # new ID
                        newnet_id += 1
                    else:
                        for neweid, rset in newent_reset[e1_old].items():
                            if r_union == rset:
                                e1_new = neweid
                                temp_ent_old2list_dict[e2_old].append(e2_new)
                                break

                if r_union != oldent_rset[e2_old]:
                    if r_union not in newent_reset[e2_old].values():
                        e2_new = newnet_id
                        newentid_dict[e2_new] = e2_old
                        ent_neigh_dict[e2_new] = e2_union_list
                        newent_reset[e2_old][e2_new] = r_union
                        temp_ent_old2list_dict[e2_old].append(e2_new)
                        kg2_ent_new.append(newnet_id) #
                        newnet_id += 1
                    else:
                        for neweid, rset in newent_reset[e2_old].items():
                            if r_union == rset:
                                e2_new = neweid
                                temp_ent_old2list_dict[e2_old].append(e2_new)
                                break

                temp_pairs_union_dict[(e1_old, e2_old)] = (e1_new, e2_new, e1_union_list, e2_union_list)
            ent_new_pair[(e1_old, e2_old)] = (e1_new, e2_new)

    assert len(newentid_dict) == newnet_id

    return newentid_dict, ent_neigh_dict, ent_new_pair, kg1_ent_new, kg2_ent_new, temp_pairs_union_dict, temp_ent_old2list_dict
