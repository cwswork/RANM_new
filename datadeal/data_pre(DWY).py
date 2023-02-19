import json
import math
import os
import re
import time
import random
import numpy as np

from autil import fileUtil
from datadeal import pre_embeds

def read_relation_triples(file_path):
    '''
    read relation_triples
    :param file_path: rel_triples_1
    :return: triples, entities, relations(h, r, t)
    '''
    print("read relation triples:", file_path)
    if file_path is None:
        return set(), set(), set()
    triples = set()
    entities, relations = set(), set()
    file = open(file_path, 'r', encoding='utf8')
    for line in file.readlines():
        params = line.strip('\n').split('\t')
        assert len(params) == 3
        h = int(params[0].strip())
        r = int(params[1].strip())
        t = int(params[2].strip())
        triples.add((h, r, t))
        entities.add(h)
        entities.add(t)
        relations.add(r)
    print("Number of entities:", len(entities))
    print("Number of relations:", len(relations))
    print("Number of relation triples:", len(triples))
    return triples, entities, relations


def read_entity_dict(file):
    print('loading a ent_ids...' + file)
    kg_ent2id_dict = dict()
    with open(file, encoding='utf-8') as f:
        for line in f:
            th = line[:-1].split('\t')
            kg_ent2id_dict[th[1]] = int(th[0])  # (name:eid)

    return kg_ent2id_dict

########################
def set_relation2id_byName(datasetPath, ordered=False):
    kg1_rel_triples, kg1_entities, kg1_relations = fileUtil.read_relation_triples(
        datasetPath + 's_triples')  # KG1
    kg2_rel_triples, kg2_entities, kg2_relations = fileUtil.read_relation_triples(
        datasetPath + 't_triples')  # kg2

    # entity #
    kg1_ent2id_dict, kg2_ent2id_dict = fileUtil.gen_mapping_id(kg1_rel_triples, kg1_entities, kg2_rel_triples,
                                                         kg2_entities, ordered=ordered)
    ent_dict = dict(kg1_ent2id_dict, **kg2_ent2id_dict)
    assert len(kg1_ent2id_dict) + len(kg2_ent2id_dict) == len(ent_dict)

    fileUtil.save_dict2txt(datasetPath + 'pre4/ent_dict_old', ent_dict, save_kv='vk')  # ent_dict{name:id)
    fileUtil.save_dict2txt(datasetPath + 'pre4/kg1_ent_dict_old', kg1_ent2id_dict, save_kv='vk')
    fileUtil.save_dict2txt(datasetPath + 'pre4/kg2_ent_dict_old', kg2_ent2id_dict, save_kv='vk')


    # relation
    kg1_rel_dict, kg2_rel_dict = fileUtil.gen_mapping_id(kg1_rel_triples, kg1_relations, kg2_rel_triples,
                                                         kg2_relations, ordered=ordered)
    fileUtil.save_dict2txt(datasetPath + 'pre4/kg1_rel_dict', kg1_rel_dict, save_kv='vk')   # ent_dict{name:id)
    fileUtil.save_dict2txt(datasetPath + 'pre4/kg2_rel_dict', kg2_rel_dict, save_kv='vk')

    #  rel_triples_id
    kg1_rel_triples_id = fileUtil.relation_triple2ids(kg1_rel_triples, kg1_ent2id_dict, kg1_rel_dict)  # 格式是(h, r, t)
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


def get_entity_embed2(datasetPath):
    print("load file: vectorList.json")
    with open(file=datasetPath + 'vectorList.json', mode='r', encoding='utf-8') as f:
        embedding_list = json.load(f)
        print(len(embedding_list), 'rows,', len(embedding_list[0]), 'columns.')

    entity_embedding = np.array(embedding_list)
    fileUtil.savepickle(datasetPath + 'pre/entity_embedding.out', entity_embedding)  # ndarray(30000,300) ,按两个kg的e编号


def set_links_file(datasetPath, dataset_division):

    ILL = fileUtil.load_list(datasetPath + 'ent_links')
    ILL_len = len(ILL)  # illL=15000
    np.random.shuffle(ILL)
    train_links_id = np.array(ILL[:ILL_len // 10 * 3])  # 30%
    test_links_id = ILL[ILL_len // 10 * 3:]  # 70%

    fileUtil.save_links(datasetPath + dataset_division + 'train_links', train_links_id)
    fileUtil.save_links(datasetPath + dataset_division + 'test_links', test_links_id)
    print('save files...train_links_id...')


def set_links_ID(datasetPath, dataset_division):
    kg1_entity2index = fileUtil.load_ids2dict(datasetPath + 'pre4/kg1_ent_dict', read_kv='vk')
    kg2_entity2index = fileUtil.load_ids2dict(datasetPath + 'pre4/kg2_ent_dict', read_kv='vk')

    train_links_ID = fileUtil.get_links_ids(datasetPath + dataset_division + 'train_links',
                                              kg1_entity2index, kg2_entity2index)
    test_links_ID = fileUtil.get_links_ids(datasetPath + dataset_division + 'test_links',
                                             kg1_entity2index, kg2_entity2index)
    # valid_links_ID = fileUtil.get_links_ids(datasetPath + dataset_division + 'valid_links',
    #                                           kg1_entity2index, kg2_entity2index)
    print('save files...train_links...')
    fileUtil.save_list2txt(datasetPath + dataset_division + 'train_links_id', train_links_ID)
    fileUtil.save_list2txt(datasetPath + dataset_division + 'test_links_id', test_links_ID)
    #fileUtil.save_list2txt(datasetPath + dataset_division + 'valid_links_id', valid_links_ID)


###################
def set_relation2id(datasetPath):
    print()
    kg1_ent_dict = read_entity_dict(datasetPath + 'pre4/kg1_ent_dict')  # (name:eid)
    kg2_ent_dict = read_entity_dict(datasetPath + 'pre4/kg2_ent_dict')  # en
    KG1_E = len(kg1_ent_dict)
    KG2_E = len(kg2_ent_dict)
    print("Num of KG1 entitys:", KG1_E)
    print("Num of KG2 entitys:", KG2_E)
    ent_dict = dict(kg1_ent_dict, **kg2_ent_dict)
    assert len(kg1_ent_dict) + len(kg1_ent_dict) == len(ent_dict)

    fileUtil.save_dict2txt(datasetPath + 'pre/ent_dict', ent_dict)
    fileUtil.save_dict2txt(datasetPath + 'pre/kg1_ent_dict', kg1_ent_dict)
    fileUtil.save_dict2txt(datasetPath + 'pre/kg2_ent_dict', kg2_ent_dict)

    kg1_rel_triples_id, kg1_entities_id, kg1_relations_id = read_relation_triples(datasetPath + 'triples_1')
    kg2_rel_triples_id, kg2_entities_id, kg2_relations_id = read_relation_triples(datasetPath + 'triples_2')
    print("Num of KG1 relations:", len(kg1_relations_id))
    print("Num of KG2 relations:", len(kg2_relations_id))
    print("Num of KG1 relation triples:", len(kg1_rel_triples_id))
    print("Num of KG2 relation triples:", len(kg2_rel_triples_id))

    rel_triples_id = list(kg1_rel_triples_id) + list(kg2_rel_triples_id)
    print('rel_triples_id:', len(rel_triples_id))
    rel_triples_id = list(set(rel_triples_id))
    print('rel_triples_id:', len(rel_triples_id))
    fileUtil.save_list2txt(datasetPath + 'pre/rel_triples_id', rel_triples_id)

    KG_E = len(kg1_ent_dict) + len(kg2_ent_dict)
    KG_R = len(set(kg1_relations_id.union(kg2_relations_id)))

    return rel_triples_id, ent_dict, kg1_ent_dict, kg2_ent_dict, KG_E, KG_R


if __name__ == '__main__':
    print("start==" + time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
    # 参DWY100k(NMN)/dbp_wd/, DWY100k(NMN)/dbp_yg
    datasetPath = '../../0315datasets/DWY100K/dbp_wd/'
    dataset_division = '721_5fold/1/'
    seed = 72  # seed =72, 3, 26, 728, 20
    print(datasetPath)
    random.seed(seed)
    np.random.seed(seed)
    # 创建目录
    # if not os.path.exists(datasetPath + 'pre4/'):
    #     print('dir not exists：' + 'pre4/')
    #     os.makedirs(datasetPath + 'pre4/')

    #####从文件夹读取kg文件：训练数据集目录
    #set_relation2id_byName(datasetPath)

    # 获得实体名
    #get_entity_embed2(datasetPath)

    # if 'wd' in datasetPath:
    #     kg2_ent_dict = fileUtil.load_dict(datasetPath + 'pre/kg2_ent_dict', read_kv='kv')
    #     entity_local_name = fileUtil.load_dict(datasetPath + 'pre/entity_local_name_2', read_kv='kv') # (name:name_new)
    #     for eid, ename in kg2_ent_dict.items():
    #         ename = entity_local_name[ename]
    #         kg2_ent_dict[eid] = ename
    #
    #     fileUtil.save_dict2txt(datasetPath + 'pre/kg2_ent_dict_new', kg2_ent_dict, save_mode='kv')

    # if not os.path.exists(datasetPath + dataset_division):
    #     print('dir not exists：' + dataset_division)
    #     os.makedirs(datasetPath + dataset_division)
    # set_links_file(datasetPath, dataset_division)
    set_links_ID(datasetPath, dataset_division)

    print("end==" + time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
