import json
import math
import os
import re
import time
import random
import numpy as np

from autil import fileUtil
from datadeal import pre_embeds


def set_relation2id_byID(datasetPath):
    #
    kg1_ent2id_dict = fileUtil.load_ids2dict(datasetPath + 'ent_ids_1', read_kv='vk')  # (name:eid) - id:name
    kg2_ent2id_dict = fileUtil.load_ids2dict(datasetPath + 'ent_ids_2', read_kv='vk')  # en
    fileUtil.save_dict2txt(datasetPath + 'pre4/kg1_ent_dict', kg1_ent2id_dict, save_kv='vk')
    fileUtil.save_dict2txt(datasetPath + 'pre4/kg2_ent_dict', kg2_ent2id_dict, save_kv='vk')

    print("Num of KG1 entitys:", len(kg1_ent2id_dict))
    print("Num of KG2 entitys:", len(kg2_ent2id_dict))

    #  ######
    kg1_rel_triples_id, kg1_entities_id, kg1_relations_id = fileUtil.read_relation_triples(datasetPath + 'triples_1')
    kg2_rel_triples_id, kg2_entities_id, kg2_relations_id = fileUtil.read_relation_triples(datasetPath + 'triples_2')
    print("Num of KG1 relations:", len(kg1_relations_id))
    print("Num of KG2 relations:", len(kg2_relations_id))
    print("Num of KG1 relation triples:", len(kg1_rel_triples_id))
    print("Num of KG2 relation triples:", len(kg2_rel_triples_id))

    rel_triples_id = list(kg1_rel_triples_id) + list(kg2_rel_triples_id)
    print('rel_triples_id:', len(rel_triples_id))
    rel_triples_id = list(set(rel_triples_id))
    print('rel_triples_id set :', len(rel_triples_id))
    fileUtil.save_triple2txt(datasetPath + 'pre4/rel_triples_id', rel_triples_id)
    #
    KG_E = len(kg1_ent2id_dict) + len(kg2_ent2id_dict)
    KG_R = len(kg1_relations_id) + len(kg2_relations_id)
    print()
    print("Num of KGs entitys:", KG_E)
    print("Num of KGs relations:", KG_R)
    print("Num of KGs relation triples:", len(rel_triples_id))

    with open(datasetPath + 'pre4/kgs_num', 'w') as ff:
        ff.write('KG_E:' + str(KG_E) + '\n')
        ff.write('KG_R:' + str(KG_R) + '\n')


def set_relation2id_byName(datasetPath, ordered=False):
    kg1_rel_triples, kg1_entities, kg1_relations = fileUtil.read_relation_triples(
        datasetPath + 's_triples')  # KG1
    kg2_rel_triples, kg2_entities, kg2_relations = fileUtil.read_relation_triples(
        datasetPath + 't_triples')  # kg2

    #  #
    kg1_ent2id_dict, kg2_ent2id_dict = fileUtil.gen_mapping_id(kg1_rel_triples, kg1_entities, kg2_rel_triples,
                                                         kg2_entities, ordered=ordered)
    ent_dict = dict(kg1_ent2id_dict, **kg2_ent2id_dict)
    assert len(kg1_ent2id_dict) + len(kg2_ent2id_dict) == len(ent_dict)

    fileUtil.save_dict2txt(datasetPath + 'pre4/ent_dict_old', ent_dict, save_kv='vk')  # ent_dict{name:id)
    fileUtil.save_dict2txt(datasetPath + 'pre4/kg1_ent_dict_old', kg1_ent2id_dict, save_kv='vk')
    fileUtil.save_dict2txt(datasetPath + 'pre4/kg2_ent_dict_old', kg2_ent2id_dict, save_kv='vk')


    #
    kg1_rel_dict, kg2_rel_dict = fileUtil.gen_mapping_id(kg1_rel_triples, kg1_relations, kg2_rel_triples,
                                                         kg2_relations, ordered=ordered)
    fileUtil.save_dict2txt(datasetPath + 'pre4/kg1_rel_dict', kg1_rel_dict, save_kv='vk')   # ent_dict{name:id)
    fileUtil.save_dict2txt(datasetPath + 'pre4/kg2_rel_dict', kg2_rel_dict, save_kv='vk')

    # rel_triples_id
    kg1_rel_triples_id = fileUtil.relation_triple2ids(kg1_rel_triples, kg1_ent2id_dict, kg1_rel_dict)  # (h, r, t)
    kg2_rel_triples_id = fileUtil.relation_triple2ids(kg2_rel_triples, kg2_ent2id_dict, kg2_rel_dict)
    rel_triples_id = kg1_rel_triples_id + kg2_rel_triples_id
    fileUtil.save_triple2txt(datasetPath + 'pre4/rel_triples_id_old', rel_triples_id)

    #
    KG_E = len(kg1_ent2id_dict) + len(kg2_ent2id_dict)
    KG_R = len(kg1_rel_dict) + len(kg2_rel_dict)
    print()
    print("Num of KGs entitys:", KG_E)
    print("Num of KGs relations:", KG_R)
    print("Num of KGs relation triples:", len(rel_triples_id))

    with open(datasetPath + 'pre4/kgs_num', 'w') as ff:
        ff.write('KG_E\t' + str(KG_E) + '\n')
        ff.write('KG_R\t' + str(KG_R) + '\n')


def set_newID_dbp(datasetPath):
    kg1_ent_dict_old = fileUtil.load_ids2dict(datasetPath + 'pre4/kg1_ent_dict_old', read_kv='vk')
    kg2_ent_dict_old = fileUtil.load_ids2dict(datasetPath + 'pre4/kg2_ent_dict_old', read_kv='vk')

    kg1_ent2id_dict = fileUtil.load_ids2dict(datasetPath + 'ent_ids_1', read_kv='vk')  # (name:eid) - id:name
    kg2_ent2id_dict = fileUtil.load_ids2dict(datasetPath + 'ent_ids_2', read_kv='vk')  # en

    oldid2newid_dict = dict()
    for name, id_old in kg1_ent_dict_old.items():
        oldid2newid_dict[id_old] = kg1_ent2id_dict[name]

    for name, id_old in kg2_ent_dict_old.items():
        oldid2newid_dict[id_old] = kg2_ent2id_dict[name]

    ent_dict = dict(kg1_ent2id_dict, **kg2_ent2id_dict)
    assert len(kg1_ent2id_dict) + len(kg2_ent2id_dict) == len(ent_dict)
    fileUtil.save_dict2txt(datasetPath + 'pre4/ent_dict', ent_dict, save_kv='vk')  # ent_dict{name:id)
    fileUtil.save_dict2txt(datasetPath + 'pre4/kg1_ent_dict', kg1_ent2id_dict, save_kv='vk')
    fileUtil.save_dict2txt(datasetPath + 'pre4/kg2_ent_dict', kg2_ent2id_dict, save_kv='vk')

    rel_triples_id = fileUtil.load_triples_id(datasetPath + 'pre4/rel_triples_id_old')
    rel_triples_id_new = []
    for h,r, t in rel_triples_id:
        rel_triples_id_new.append((oldid2newid_dict[h], r, oldid2newid_dict[t]))

    fileUtil.save_triple2txt(datasetPath + 'pre4/rel_triples_id', rel_triples_id_new)


def get_entity_embed(datasetPath):
    kg1_ent_list = fileUtil.load_ids2list(datasetPath + 'pre4/ent_ids_1-trans')
    kg2_ent_list = fileUtil.load_ids2list(datasetPath + 'ent_ids_2')
    kg_ent_list = kg1_ent_list + kg2_ent_list

    kg_ent_list = sorted(kg_ent_list, key= lambda x:x[0], reverse=False)
    a = kg_ent_list[-1]
    new_entity_name_list, entity_embedding = pre_embeds.get_entity_embed(kg_ent_list)
    fileUtil.save_list2txt(datasetPath + 'pre4/ent_dict_replace_name.txt', new_entity_name_list)
    fileUtil.savepickle(datasetPath + 'pre4/entity_embedding.out', entity_embedding)  # ndarray(30000,300)

def get_entity_embed_100K(datasetPath):
    print("load file: vectorList.json")
    with open(file=datasetPath + 'vectorList.json', mode='r', encoding='utf-8') as f:
        embedding_list = json.load(f)
        print(len(embedding_list), 'rows,', len(embedding_list[0]), 'columns.')

    entity_embedding = np.array(embedding_list)
    fileUtil.savepickle(datasetPath + 'pre4/entity_embedding.out', entity_embedding)  # ndarray(30000,300)


def set_links_file(datasetPath, dataset_division):
    ILL = fileUtil.load_triples_id(datasetPath + 'ref_ent_ids')
    kg1_ent2id_dict = fileUtil.load_ids2dict(datasetPath + 'ent_ids_1', read_kv='kv')
    kg2_ent2id_dict = fileUtil.load_ids2dict(datasetPath + 'ent_ids_2', read_kv='kv')
    IIL_ids = []
    for e1, e2 in ILL:
        IIL_ids.append((kg1_ent2id_dict[e1], kg2_ent2id_dict[e2]))

    ILL_len = len(IIL_ids)  # illL=15000
    np.random.shuffle(IIL_ids)
    train_links = np.array(IIL_ids[:ILL_len // 10 * 2])  # 20%
    valid_links = np.array(IIL_ids[len(train_links):ILL_len // 10 * 3])  # 10%
    test_links = IIL_ids[ILL_len // 10 * 3:]  # 70%

    print('save files...train_links...')
    fileUtil.save_list2txt(datasetPath + dataset_division + 'train_links', train_links)
    fileUtil.save_list2txt(datasetPath + dataset_division + 'valid_links', valid_links)
    fileUtil.save_list2txt(datasetPath + dataset_division + 'test_links', test_links)


def set_links_file_100K(datasetPath, dataset_division):
    ent_links = fileUtil.load_list(datasetPath + 'ent_links')

    ILL_len = len(ent_links)  # illL=15000
    np.random.shuffle(ent_links)
    train_links = np.array(ent_links[:ILL_len // 10 * 3])  # 30%
    test_links = ent_links[ILL_len // 10 * 3:]  # 70%

    print('save files...train_links...')
    fileUtil.save_list2txt(datasetPath + dataset_division + 'train_links', train_links)
    fileUtil.save_list2txt(datasetPath + dataset_division + 'test_links', test_links)


def set_links_ID(datasetPath, dataset_division):
    kg1_entity2index = fileUtil.load_ids2dict(datasetPath + 'ent_ids_1', read_kv='vk')
    kg2_entity2index = fileUtil.load_ids2dict(datasetPath + 'ent_ids_2', read_kv='vk')

    train_links_ID = fileUtil.get_links_ids(datasetPath + dataset_division + 'train_links',
                                              kg1_entity2index, kg2_entity2index)
    test_links_ID = fileUtil.get_links_ids(datasetPath + dataset_division + 'test_links',
                                             kg1_entity2index, kg2_entity2index)
    valid_links_ID = fileUtil.get_links_ids(datasetPath + dataset_division + 'valid_links',
                                              kg1_entity2index, kg2_entity2index)
    print('save files...train_links...')
    fileUtil.save_list2txt(datasetPath + dataset_division + 'train_links_id', train_links_ID)
    fileUtil.save_list2txt(datasetPath + dataset_division + 'valid_links_id', valid_links_ID)
    fileUtil.save_list2txt(datasetPath + dataset_division + 'test_links_id', test_links_ID)

if __name__ == '__main__':
    print("start==" + time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))

    # fr_en、ja_en、zh_en   DWY100K/dbp_wd/ /dbp_yg
    datasetPath = '../../0315datasets/zh_en(dbp15)/'
    # datasetPath = '../../0315datasets/DWY100K/dbp_yg/'
    dataset_division = '721_5fold/1/'
    seed = 72  # seed =72, 3, 26, 728, 20
    print(datasetPath)
    random.seed(seed)
    np.random.seed(seed)
    if not os.path.exists(datasetPath + dataset_division):
        print('dir not exists：' + dataset_division)
        os.makedirs(datasetPath + dataset_division)

    set_links_ID(datasetPath, dataset_division)

    ##从文件夹读取kg文件：训练数据集目录
    #set_relation2id_byName(datasetPath)
    # 新旧ID变换
    #set_newID_dbp(datasetPath)
    # 获得实体名嵌入
    # get_entity_embed_100K(datasetPath)
    # 生成links_ID文件
    # set_links_file_100K(datasetPath, dataset_division)

    print("end==" + time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
