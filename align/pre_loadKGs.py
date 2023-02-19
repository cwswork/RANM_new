import os
import torch
import numpy as np

from autil import fileUtil
from align import pre_candidate, pre_relScore

class load_KGs_data(object):
    def __init__(self, myconfig):
        # Load Datasets
        kgs_num_dict = fileUtil.load_dict(myconfig.datasetPath + 'kgs_num', read_kv='kv')
        self.kg_E = int(kgs_num_dict['KG_E'])  # KG_E
        self.kg_R = int(kgs_num_dict['KG_R'])  # KG_R
        # self.kg_M = int(kgs_num_dict['KG_M'])  # KG_M
        # self.kg_V = int(kgs_num_dict['KG_V'])  # KG_V

        kg1_entity2index = fileUtil.load_ids2dict(myconfig.datasetPath + 'kg1_ent_dict', read_kv='vk') # name:id
        kg2_entity2index = fileUtil.load_ids2dict(myconfig.datasetPath + 'kg2_ent_dict', read_kv='vk')
        self.kg1_ent = list(kg1_entity2index.values())
        self.kg2_ent = list(kg2_entity2index.values())
        # kg1_index2entity = {e: idx for idx, e in kg1_entity2index.items()}

        ## Relation triple
        self.rel_triple = fileUtil.load_triples_id(myconfig.datasetPath + 'rel_triples_id')
        ## entity embedding
        self.ename_embed = fileUtil.loadpickle(myconfig.datasetPath + 'entity_embedding.out')
        self.ename_embed = torch.FloatTensor(self.ename_embed)

        ## train、Valid、Test  # np.array
        if '100' not in myconfig.datasetPath:
            self.train_links = fileUtil.get_links_ids(
                myconfig.datasetPath[:-5] + myconfig.dataset_division + 'train_links', kg1_entity2index, kg2_entity2index)
            self.test_links = fileUtil.get_links_ids(
                myconfig.datasetPath[:-5] + myconfig.dataset_division + 'test_links', kg1_entity2index, kg2_entity2index)
            self.valid_links = fileUtil.get_links_ids(myconfig.datasetPath[:-5] + myconfig.dataset_division + 'valid_links',
                                             kg1_entity2index, kg2_entity2index)
        else:
            self.valid_links = []
            self.train_links = fileUtil.get_links_ids2(myconfig.datasetPath[:-5] + myconfig.dataset_division + 'train_links_id')
            self.test_links = fileUtil.get_links_ids2(myconfig.datasetPath[:-5] + myconfig.dataset_division + 'test_links_id')

        self.pre_reldata(self.ename_embed, myconfig)  #


    # Pre-processing before training
    def pre_reldata(self, ent_embed, myconfig):
        save_temp = myconfig.division_save + 'temp/'
        if not os.path.exists(save_temp):
            os.makedirs(save_temp)

        #1. Obtain candidate entity pairs for training set, test set, etc. ——ent_neigh_dict
        ent_neigh_dict = pre_relScore.ent_neigh_gene(self.rel_triple, self.kg_E)  # (30000, ?)
        myconfig.myprint('number of ent_neigh_dict:' + str(len(ent_neigh_dict)))

        #2. Calculate the number of ratings for the entity pairs in the training set ——left_neigh_match
        max_neighbors_num = len(max(ent_neigh_dict.values(), key=lambda x: len(x)))  #
        myconfig.myprint('Maximum number of neighbors:' + str(max_neighbors_num)) # max_neighbors_num = 235
        ent_pad_id, rel_pad_id = self.kg_E, self.kg_R #
        left_neigh_match = pre_relScore.neigh_match(ent_embed, self.train_links, ent_neigh_dict,
                                                    max_neighbors_num, ent_pad_id, rel_pad_id, myconfig)
        # [B,neigh,5]: (tail_i, tail_j, rel_i, rel_j, ent_ij_sim)

        #3. According to the number of scores of the entity pairs in the training set, determine the relationship pair matching  ——RR_pair_dict
        ent_pad_id = self.kg_E
        RR_pair_dict, temp_RR_list, temp_notRR_list = pre_relScore.rel_match(left_neigh_match, ent_pad_id, myconfig)
        myconfig.myprint("Number of RRpair_dict:" + str(len(RR_pair_dict)))
        fileUtil.save_list2txt(save_temp + 'RR_list.txt', temp_RR_list)
        fileUtil.save_list2txt(save_temp + 'notRR_list.txt', temp_notRR_list)

        #4. Align according to the R name
        kg1_index2rel = fileUtil.load_ids2dict(myconfig.datasetPath + 'kg1_rel_dict', read_kv='kv')  # id:name
        kg2_index2rel = fileUtil.load_ids2dict(myconfig.datasetPath + 'kg2_rel_dict', read_kv='kv')
        Rname_dict = pre_relScore.get_Rname_dict(kg1_index2rel, kg2_index2rel)
        myconfig.myprint('Number of Rname_dict:' + str(len(Rname_dict)))

        for (r1_id, r2_id) in Rname_dict.items():
            if r1_id not in RR_pair_dict.keys() and r2_id not in RR_pair_dict.values():
                RR_pair_dict[r1_id] = r2_id
        fileUtil.save_dict2txt(save_temp + 'RR_pair_dict_new.txt', RR_pair_dict)
        myconfig.myprint("Number of RRpair_dict after Rname_dict:" + str(len(RR_pair_dict)))

        #5. According to RR_pair_dict, update the aligned Relation ID
        Rold2new_dict, self.rel_triple, ent_neigh_dict, Rnew_num = pre_relScore.rese_relid(RR_pair_dict, kg1_index2rel, kg2_index2rel,
                                        self.rel_triple, ent_neigh_dict)
        myconfig.myprint('Number of old Relation:' + str(self.kg_R))
        myconfig.myprint('Number of new Relation:' + str(Rnew_num))

        #self.RR_pair_dict = RR_pair_dict #
        self.ent_neigh_dict = ent_neigh_dict  #   (kg_E_new, kg_E)
        self.kg_R = Rnew_num

    # During the training process, new entity preprocessing
    def pre_gen(self, ent_embed, myconfig):
        save_temp = myconfig.division_save + 'temp/'
        # 6. Generate candidate set——ent_candidates
        candidate_num_1 = 100
        myconfig.myprint("get " + str(candidate_num_1) + " candidate by " + myconfig.metric + " similartity.")
        ent_candidates = pre_candidate.gen_candidate(ent_embed, self.train_links, self.valid_links, self.test_links,
                                candidate_num=candidate_num_1, metric=myconfig.metric, is_cuda=myconfig.is_cuda)  # dict(15000: list(100))
        # 7. Obtain matching neighbors of candidate entity pairs
        temp_rel_match_dict, temp_ent_rset = pre_candidate.get_match_rel(self.ent_neigh_dict,
                                                                        ent_candidates, self.kg1_ent, self.kg2_ent, myconfig) #

        # 8. Get a new temporary neighbor relationship —— newentid_dict
        newentid_dict, ent_neigh_dict, ent_new_pair, kg1_ent_new, kg2_ent_new\
            , temp_pairs_union_dict, ent_old2list_dict = pre_candidate.get_candidate_new(ent_candidates, self.ent_neigh_dict, temp_rel_match_dict,
                                                                      temp_ent_rset, self.kg_E, self.kg1_ent, self.kg2_ent)

        with open(save_temp + 'temp_pairs_union_dict.txt', 'w', encoding='utf-8') as fw:
            fw.write('(节点1\t节点2)\t(节点1新ID\t节点2新ID)\t节点1共有邻居\t节点2共有邻居\n')
            # temp_pairs_union_dict[(e1_old, e2_old)] = (e1_new, e2_new, e1_union_list, e2_union_list)
            for (e1_old, e2_old), (e1_new, e2_new, e1_union_list, e2_union_list) in temp_pairs_union_dict.items():
                fw.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(e1_old, e2_old, e1_new, e2_new,
                                  e1_union_list.__str__(), e2_union_list.__str__()))

        myconfig.myprint('Number of newentid_dict:' + str(len(newentid_dict)))
        myconfig.myprint('Number of ent_neigh_dict:' + str(len(ent_neigh_dict)))  # ent_neigh_dict
        myconfig.myprint('Number of ent_new_pair:' + str(len(ent_new_pair)))
        myconfig.myprint('Number of temp_pairs_union_dict:' + str(len(temp_pairs_union_dict)))
        myconfig.myprint('Number of kg1_ent_new:' + str(len(kg1_ent_new)))
        myconfig.myprint('Number of kg2_ent_new:' + str(len(kg2_ent_new)))

        self.newentid_dict = newentid_dict  #
        self.ent_neigh_dict = ent_neigh_dict  #
        self.ent_new_pair = ent_new_pair  #
        self.ent_old2list_dict = ent_old2list_dict

        return newentid_dict, ent_neigh_dict, ent_new_pair, ent_old2list_dict



