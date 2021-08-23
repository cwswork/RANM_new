import copy
import time
import torch
import numpy as np

from autil import alignment


#1. Obtain candidate entity pairs for training set, test set, etc. ——ent_neigh_dict
def ent_neigh_gene(rel_triples, kg_E):
    """
    get one hop neighbor of entity
    return a dict, key = entity, value = (padding) neighbors of entity
    """
    ent_neigh_dict = dict()
    for eid in range(kg_E):
        ent_neigh_dict[eid] = []

    for (h, r, t) in rel_triples:
        ent_neigh_dict[h].append((t, r))
        # if h == t: ??
        #     continue
        ent_neigh_dict[t].append((h, r))

    return ent_neigh_dict


#2. Calculate the number of ratings for the entity pairs in the training set ——left_neigh_match
def neigh_match(ename_embed, train_links_array, ent_neigh_dict_old, max_neighbors_num, ent_pad_id, rel_pad_id, myconfig):
    """Similarity Score (between entity pairs)
        return: [B,neigh,5]: (tail_i, tail_j, rel_i, rel_j, ent_ij_sim)
    """
    ent_neigh_dict = copy.deepcopy(ent_neigh_dict_old)
    for e in ent_neigh_dict.keys():
        pad_list = [(ent_pad_id, rel_pad_id)] * (max_neighbors_num - len(ent_neigh_dict[e]))
        ent_neigh_dict[e] += pad_list

    if ename_embed.is_cuda:
        ename_embed = ename_embed.detach().cpu().numpy()
    else:
        ename_embed = ename_embed.detach().numpy()
    dim = len(ename_embed[0])
    zero_embed = [0.0 for _ in range(dim)]  # <PAD> embedding
    ename_embed = np.vstack([ename_embed, zero_embed])
    ename_embed = torch.FloatTensor(ename_embed)

    myconfig.myprint("train_pairs (e1,e2) num is: {}".format(len(train_links_array)))
    start_time = time.time()
    left_neigh_match, right_neigh_match = [], []
    batch_size = 3000
    for start_pos in range(0, len(train_links_array), batch_size):  # len(ent_pairs)=750000
        end_pos = min(start_pos + batch_size, len(train_links_array))
        batch_ent_pairs = train_links_array[start_pos: end_pos]
        e1s = [e1 for e1, e2 in batch_ent_pairs]
        e2s = [e2 for e1, e2 in batch_ent_pairs]
        #(t, r)
        er1_neigh = np.array([ent_neigh_dict[e1] for e1 in e1s])  # size: [B(Batchsize),ne1(e1_neighbor_max_num)]
        er2_neigh = np.array([ent_neigh_dict[e2] for e2 in e2s])
        e1_neigh = er1_neigh[:, :, 0]  # e
        e2_neigh = er2_neigh[:, :, 0]
        r1_neigh = er1_neigh[:, :, 1]  # r
        r2_neigh = er2_neigh[:, :, 1]

        e1_neigh_tensor = torch.LongTensor(e1_neigh) # [B,neigh]
        e2_neigh_tensor = torch.LongTensor(e2_neigh)
        e1_neigh_emb = ename_embed[e1_neigh_tensor]  # [B,neigh,embedding_dim]
        e2_neigh_emb = ename_embed[e2_neigh_tensor]

        max_index, max_scoce = alignment.torch_sim_max_single(e1_neigh_emb, e2_neigh_emb,
                                               batch_size=len(e1_neigh_emb), top_num=1, metric='cosine')
        max_scoce = max_scoce.squeeze(-1)  # [B,neigh,1] -> [B,neigh] #get max value.
        max_index = max_index.squeeze(-1)

        batch_match = np.zeros([e1_neigh.shape[0], e1_neigh.shape[1], 5])
        for e in range(e1_neigh.shape[0]): # [B,neigh]
            e1_array = e1_neigh[e] # [neigh,1] = >[neigh]
            e2_array = e2_neigh[e, max_index[e]]
            r1_array = r1_neigh[e]
            r2_array = r2_neigh[e, max_index[e]]
            scoce_array = max_scoce[e]
            if len(e2_array) != len(set(e2_array.tolist())): # Many to one
                for i in range(1, len(e2_array)):
                    if e1_array[i] == ent_pad_id: # Empty neighbor
                        break
                    if e2_array[i] in e2_array[0: i]:
                        index = np.where(e2_array[0: i] == e2_array[i])[0][0]
                        if scoce_array[index] > scoce_array[i]:
                            e2_array[i] = ent_pad_id
                        else:
                            e2_array[index] = ent_pad_id
            aa = np.vstack((e1_array, e2_array, r1_array, r2_array, scoce_array)).T
            batch_match[e] = aa

        if type(left_neigh_match) is np.ndarray:
            left_neigh_match = np.vstack((left_neigh_match, batch_match))
        else:
            left_neigh_match = batch_match  # [B,neigh,5]: (tail_i, tail_j, rel_i, rel_j, ent_ij_sim)

    myconfig.myprint("all ent pair left_neigh_match shape:{}".format(left_neigh_match.shape) )
    myconfig.myprint("using time {:.3f}".format(time.time() - start_time))
    return left_neigh_match


#3. According to the number of scores of the entity pairs in the training set, determine the relationship pair matching ——RR_pair_dict
def rel_match(left_neigh_match, ent_pad_id, myconfig):
    # left_neigh_match [B,neigh,5]: (tail_i, tail_j, rel_i, rel_j, ent_ij_sim)

    rel_pair_score = {}
    ent_ij_sim5_num = 0
    for neigh_ll in left_neigh_match.tolist():
        for (tail_i, tail_j, rel_i, rel_j, ent_ij_sim) in neigh_ll:
            tail_i, tail_j, rel_i, rel_j = int(tail_i), int(tail_j), int(rel_i), int(rel_j)
            if tail_i == ent_pad_id or tail_j == ent_pad_id or ent_ij_sim == 0.0:
                continue

            if ent_ij_sim < 0.5: # Exclude low similarity
                ent_ij_sim5_num += 1
                continue
            if (rel_i, rel_j) not in rel_pair_score.keys():
                rel_pair_score[(rel_i, rel_j)] = [ent_ij_sim, 1]
            else:
                rel_pair_score[(rel_i, rel_j)][0] += ent_ij_sim  # Similarity stacking
                rel_pair_score[(rel_i, rel_j)][1] += 1 # numbery

    for rel_pair, (score, num) in rel_pair_score.items():
        rel_pair_score[rel_pair] = [score/num, num] # Average similarity

    myconfig.myprint('ent_ij_sim5_num:' + str(ent_ij_sim5_num))
    myconfig.myprint("all rel_pair_score len:" + str(len(rel_pair_score)))
    # Sort by descending "number of matches"
    sim_rank_order = sorted(rel_pair_score.items(), key=lambda kv: kv[1][1], reverse=True)
    RR_list, notRR_list = [], []   # list([r1_id, r2_id, sim_v, num])
    RR_pair_dict = dict()
    for (r1_id, r2_id), (sim_v, num) in sim_rank_order:
        if r1_id not in RR_pair_dict.keys() and r2_id not in RR_pair_dict.values():
            if num < 10: # Exclude too few matches
                continue
            RR_pair_dict[r1_id] = r2_id
            RR_list.append([r1_id, r2_id, sim_v, num])
        else:
            notRR_list.append([r1_id, r2_id, sim_v, num])

    RR_list = sorted(RR_list, key=lambda kv: kv[0], reverse=False)
    notRR_list = sorted(notRR_list, key=lambda kv: kv[0], reverse=False)
    return RR_pair_dict, RR_list, notRR_list

#4. Align according to the R name
def get_Rname_dict(kg1_index2rel, kg2_index2rel):
    # Relation with similar names are not necessarily aligned! !
    kg1_id2rel_dict = dict()
    for rid1, rname1 in kg1_index2rel.items():
        rname1 = rname1.split('/')[-1]
        kg1_id2rel_dict[rname1] = rid1

    kg2_id2rel_dict = dict()
    for rid2, rname2 in kg2_index2rel.items():
        rname2 = rname2.split('/')[-1]
        kg2_id2rel_dict[rname2] = rid2

    Rname_dict = {}
    for rname1, rid1 in kg1_id2rel_dict.items():
        if rname1 in kg2_id2rel_dict.keys():
            Rname_dict[rid1] = kg2_id2rel_dict[rname1] # r1_d1-r2_id

    return Rname_dict


#5. According to RR_pair_dict, update the aligned Relation ID
def rese_relid(RR_pair_dict, kg1_index2rel, kg2_index2rel, rel_triple, ent_neigh_dict):
    Rold2new_dict = dict()
    # alignment relation
    rid_new = 0
    for r1, r2 in RR_pair_dict.items():
        Rold2new_dict[r1] = rid_new
        Rold2new_dict[r2] = rid_new
        rid_new += 1
    # KG1
    for rid_old in kg1_index2rel.keys():
        if rid_old not in Rold2new_dict.keys():
            Rold2new_dict[rid_old] = rid_new
            rid_new += 1
    # KG2
    for rid_old in kg2_index2rel.keys():
        if rid_old not in Rold2new_dict.keys():
            Rold2new_dict[rid_old] = rid_new
            rid_new += 1
    # update rel triple
    rel_triple_new = []
    for h, r, t in rel_triple:
        rel_triple_new.append((h, Rold2new_dict[r], t))

    # update ent_neigh_dict
    for e1, er_list in ent_neigh_dict.items():
        er_list_new = []
        for e2, r in er_list:
            er_list_new.append((e2, Rold2new_dict[r]))
        ent_neigh_dict[e1] = er_list_new

    return Rold2new_dict, rel_triple_new, ent_neigh_dict, rid_new

